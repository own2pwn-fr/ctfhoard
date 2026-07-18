"""Exhaustive GitHub *discovery* of CTF challenge-source and writeup repositories.

This module answers a single question — *which* GitHub repos are worth mirroring —
and nothing more. It is deliberately separate from extraction: :mod:`ctfhoard`'s
``git_repo`` connector walks a repo into per-challenge records, but it can only walk
repos someone hands it. Historically that list was the small curated
``seeds/official_repos.yaml``; this module grows it to the whole long tail of CTF
repositories on GitHub and emits a deduplicated ``data/discovered_repos.jsonl`` the
connector then consumes.

The hard part is GitHub's search contract, which we respect exactly:

* **1000-result cap per query.** The search API returns at most 10 pages × 100 = 1000
  results for *any* query, no matter how large ``total_count`` is. To enumerate a
  population bigger than that we shard: attach a ``created:LO..HI`` date window to the
  query and recursively **bisect** the window whenever a shard still reports
  ``total_count > 1000``, until every leaf shard fits under the cap; then page each
  leaf fully. A single day that still overflows is further bisected by ``stars:``.
* **30 requests/minute** on repository search (authenticated). Every search call goes
  through a :class:`~ctfhoard.ratelimit.RateLimiter` with a ~2 s floor, and we back off
  on ``403``/``429`` honoring ``Retry-After`` / ``x-ratelimit-reset`` and GitHub's
  secondary-rate-limit responses.
* **Query length ≤ 256 chars, ≤ 5 boolean operators.** The curated base queries plus a
  date/stars qualifier stay well under both limits.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import yaml
from loguru import logger
from pydantic import BaseModel, Field

from ctfhoard.http import make_client
from ctfhoard.ratelimit import RateLimiter

#: GitHub repository-search endpoint.
_SEARCH_URL = "https://api.github.com/search/repositories"

#: Hard per-query result ceiling imposed by the search API (10 pages × 100).
_RESULT_CAP = 1000

#: Minimum seconds between search calls to stay under 30 req/min (30/60 = 1 per 2 s).
_SEARCH_MIN_INTERVAL = 2.0

#: Earliest plausible CTF repo — GitHub predates this but CTF repos effectively start
#: around 2011; the bisection root window runs from here to "today".
_EPOCH = date(2011, 1, 1)

#: Backoff attempts on rate-limit / transient search failures before giving up.
_MAX_RETRIES = 6

#: Upper bound (seconds) we are willing to sleep for a single rate-limit backoff.
_MAX_BACKOFF = 180.0

#: Default location of the discovered-repo artifact the ``git_repo`` connector reads.
_DEFAULT_OUT = Path("data") / "discovered_repos.jsonl"

#: Path of the curated seed file, used to exclude already-known repos from discovery.
_SEEDS_PATH = Path(__file__).resolve().parents[2] / "seeds" / "official_repos.yaml"
_SEED_SECTIONS = ("official_sources", "community_archives", "writeup_archives")

#: Keywords that classify a repo as a *writeup* collection (checked first — more
#: specific than the generic "challenge" signal).
_WRITEUP_KEYWORDS = ("writeup", "write-up", "write-ups", "writeups")
#: Keywords that classify a repo as *challenge sources*.
_SOURCE_KEYWORDS = ("ctf-challenges", "challenges", "challenge", "tasks", "source")

#: Curated base queries covering both source and writeup repos. Each is fanned out
#: through :func:`sharded_search` so the union can exceed the 1000-result cap.
DEFAULT_QUERIES: tuple[str, ...] = (
    # Topic-tagged repos (highest precision).
    "topic:ctf",
    "topic:ctf-challenges",
    "topic:ctf-writeups",
    "topic:ctf-writeup",
    "topic:writeup",
    # Free-text on name/description (broad recall for untagged repos).
    "ctf writeup in:name,description",
    "ctf challenges in:name,description",
    '"ctf" in:name',
)


class RepoCandidate(BaseModel):
    """One GitHub repository discovered as a CTF source/writeup candidate.

    A thin, storage-friendly projection of the search API's repo object carrying
    exactly what downstream needs to decide whether/how to mirror it: identity, an
    activity/popularity signal (``stars`` / ``pushed_at``), licensing hint, and an
    inferred :attr:`kind` (sources vs. writeups). It is intentionally *not* a
    :class:`~ctfhoard.schema.RawChallenge` — discovery finds repos, it does not walk
    them.
    """

    full_name: str = Field(description="'owner/name' identifier on GitHub.")
    html_url: str = Field(description="Browser URL of the repository.")
    stars: int = Field(default=0, ge=0, description="Stargazer count at discovery time.")
    pushed_at: datetime | None = Field(
        default=None, description="Last push timestamp (UTC), an activity signal."
    )
    created_at: datetime | None = Field(
        default=None, description="Repository creation timestamp (UTC)."
    )
    topics: list[str] = Field(
        default_factory=list, description="GitHub topic tags attached to the repo."
    )
    description: str | None = Field(default=None, description="Repo description, if any.")
    license_spdx: str | None = Field(
        default=None,
        description="SPDX id of the repo's declared license, or None when absent / "
        "'NOASSERTION'. Only a hint — the connector re-detects at clone time.",
    )
    default_branch: str | None = Field(default=None, description="Default branch name.")
    size_kb: int = Field(default=0, ge=0, description="Repository size in kilobytes.")
    kind: str = Field(
        default="unknown",
        description="Inferred nature: 'sources', 'writeups', or 'unknown'.",
    )

    @classmethod
    def from_api(cls, item: dict) -> RepoCandidate:
        """Build a candidate from a raw search-API repository object."""
        lic = item.get("license") or {}
        spdx = lic.get("spdx_id") if isinstance(lic, dict) else None
        if spdx in (None, "NOASSERTION", ""):
            spdx = None
        topics = [t for t in (item.get("topics") or []) if isinstance(t, str)]
        name = item.get("full_name", "")
        description = item.get("description")
        return cls(
            full_name=name,
            html_url=item.get("html_url", f"https://github.com/{name}"),
            stars=int(item.get("stargazers_count") or 0),
            pushed_at=_parse_ts(item.get("pushed_at")),
            created_at=_parse_ts(item.get("created_at")),
            topics=topics,
            description=description,
            license_spdx=spdx,
            default_branch=item.get("default_branch"),
            size_kb=int(item.get("size") or 0),
            kind=infer_kind(name, description, topics),
        )


def infer_kind(name: str, description: str | None, topics: Iterable[str]) -> str:
    """Classify a repo as ``"writeups"``, ``"sources"``, or ``"unknown"``.

    Writeup signals win over source signals (a "ctf challenges writeups" repo is a
    writeup collection). Matching is substring-based over the lowercased name,
    description, and topics — cheap and robust to punctuation variants.
    """
    haystack = " ".join(
        [name.lower(), (description or "").lower(), " ".join(t.lower() for t in topics)]
    )
    if any(kw in haystack for kw in _WRITEUP_KEYWORDS):
        return "writeups"
    if any(kw in haystack for kw in _SOURCE_KEYWORDS):
        return "sources"
    return "unknown"


def _parse_ts(value: str | None) -> datetime | None:
    """Parse a GitHub ISO-8601 timestamp (``...Z``) into an aware ``datetime``."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _auth_headers(token: str | None) -> dict[str, str]:
    """Headers for an authenticated search call (topics included by default)."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_token(explicit: str | None = None) -> str | None:
    """Resolve a GitHub token from an explicit value, ``GH_TOKEN``/``GITHUB_TOKEN``,
    or the ``gh auth token`` CLI — the first that yields a non-empty string wins."""
    if explicit:
        return explicit
    for env in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    try:
        out = subprocess.run(  # noqa: S603,S607 — fixed argv, no shell
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    token = out.stdout.strip()
    return token or None


def _rate_limit_wait(resp: httpx.Response) -> float:
    """Seconds to sleep before retrying a 403/429/5xx search response.

    Honors ``Retry-After`` (secondary limits), then an exhausted
    ``x-ratelimit-remaining`` with ``x-ratelimit-reset`` (primary limit), and finally
    a conservative default. The result is clamped to :data:`_MAX_BACKOFF`.
    """
    retry_after = resp.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return min(float(retry_after), _MAX_BACKOFF)
    remaining = resp.headers.get("x-ratelimit-remaining")
    reset = resp.headers.get("x-ratelimit-reset")
    if remaining == "0" and reset and reset.isdigit():
        delta = int(reset) - int(time.time()) + 1
        return min(max(float(delta), 1.0), _MAX_BACKOFF)
    return min(60.0, _MAX_BACKOFF)


def _transient_wait(attempt: int) -> float:
    """Exponential backoff (seconds) for a network error carrying no response to read.

    Grows as ``_SEARCH_MIN_INTERVAL * 2**attempt`` and is clamped to :data:`_MAX_BACKOFF`.
    """
    return min(_SEARCH_MIN_INTERVAL * (2**attempt), _MAX_BACKOFF)


def _search_page(
    query: str,
    page: int,
    *,
    token: str | None,
    per_page: int,
    client: httpx.Client,
    limiter: RateLimiter,
    sort: str | None = None,
) -> tuple[int, list[dict]]:
    """Fetch one search page, returning ``(total_count, items)``.

    Serialized through ``limiter`` and resilient to *transient* failures: on 403/429 or
    any 5xx it backs off per :func:`_rate_limit_wait`, and on a network/timeout error it
    backs off per :func:`_transient_wait`, retrying up to :data:`_MAX_RETRIES` times.
    Non-transient client errors (e.g. 422 malformed query) still raise immediately. When
    retries are exhausted on a transient failure the page is *skipped* — returning
    ``(0, [])`` with a warning — rather than aborting the whole discovery run so that
    repos already enumerated by other shards/queries are preserved.
    """
    params: dict[str, str | int] = {"q": query, "per_page": per_page, "page": page}
    if sort:
        params["sort"] = sort
        params["order"] = "desc"
    for attempt in range(_MAX_RETRIES):
        limiter.acquire()
        try:
            resp = client.get(_SEARCH_URL, params=params, headers=_auth_headers(token))
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            wait = _transient_wait(attempt)
            logger.warning(
                "search request error ({}), backing off {:.0f}s (attempt {}/{})",
                type(exc).__name__,
                wait,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            body = resp.json()
            return int(body.get("total_count", 0)), list(body.get("items", []))
        if resp.status_code in (403, 429) or resp.status_code >= 500:
            wait = _rate_limit_wait(resp)
            logger.warning(
                "search transient failure ({}), backing off {:.0f}s (attempt {}/{})",
                resp.status_code,
                wait,
                attempt + 1,
                _MAX_RETRIES,
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
    logger.warning(
        "search page gave up after {} attempts (query={!r}, page={}); skipping",
        _MAX_RETRIES,
        query,
        page,
    )
    return 0, []


def _managed(
    client: httpx.Client | None, limiter: RateLimiter | None
) -> tuple[httpx.Client, RateLimiter, bool]:
    """Return an ``(client, limiter, owns_client)`` triple, building defaults if needed."""
    owns = client is None
    if client is None:
        client = make_client()
    if limiter is None:
        limiter = RateLimiter(min_interval=_SEARCH_MIN_INTERVAL)
    return client, limiter, owns


def search_repos(
    query: str,
    *,
    token: str | None,
    per_page: int = 100,
    client: httpx.Client | None = None,
    limiter: RateLimiter | None = None,
) -> Iterator[RepoCandidate]:
    """Yield every repo of a single search ``query``, up to the 1000-result cap.

    Pages through the query (max 10 pages) honoring the search rate limit. Callers may
    inject a shared ``client``/``limiter`` to serialize across many queries; otherwise
    a private, rate-limited client is created and closed here.
    """
    client, limiter, owns = _managed(client, limiter)
    try:
        max_pages = math.ceil(_RESULT_CAP / per_page)
        for page in range(1, max_pages + 1):
            total, items = _search_page(
                query,
                page,
                token=token,
                per_page=per_page,
                client=client,
                limiter=limiter,
                sort="updated",
            )
            for item in items:
                yield RepoCandidate.from_api(item)
            if not items or page * per_page >= min(total, _RESULT_CAP):
                break
    finally:
        if owns:
            client.close()


def _iso(d: date) -> str:
    return d.isoformat()


def _shard_by_stars(
    day_query: str,
    *,
    token: str | None,
    seen: set[str],
    client: httpx.Client,
    limiter: RateLimiter,
    lo: int = 0,
    hi: int = 1_000_000,
) -> Iterator[RepoCandidate]:
    """Last-resort bisection over ``stars:`` for a single-day shard still over the cap.

    Some very active days on ``topic:ctf`` exceed 1000 repos even within one calendar
    day; splitting the star range lets us page each sub-population fully instead of
    losing the overflow.
    """
    q = f"{day_query} stars:{lo}..{hi}"
    total, _ = _search_page(
        q, 1, token=token, per_page=1, client=client, limiter=limiter
    )
    if total == 0:
        return
    if total <= _RESULT_CAP or lo >= hi:
        if total > _RESULT_CAP:
            # Base case with no room left to split (a single star value overflows the
            # cap): we can only page the first 1000 results. Surface the loss so it is
            # observable instead of silently truncating.
            logger.warning(
                "un-splittable shard {!r} still reports {} > {} results; "
                "dropping ~{} repos beyond the search cap",
                q,
                total,
                _RESULT_CAP,
                total - _RESULT_CAP,
            )
        for cand in search_repos(q, token=token, client=client, limiter=limiter):
            if cand.full_name not in seen:
                seen.add(cand.full_name)
                yield cand
        return
    mid = lo + (hi - lo) // 2
    yield from _shard_by_stars(
        day_query, token=token, seen=seen, client=client, limiter=limiter, lo=lo, hi=mid
    )
    yield from _shard_by_stars(
        day_query, token=token, seen=seen, client=client, limiter=limiter, lo=mid + 1, hi=hi
    )


def _shard_by_dates(
    base_query: str,
    lo: date,
    hi: date,
    *,
    token: str | None,
    seen: set[str],
    client: httpx.Client,
    limiter: RateLimiter,
) -> Iterator[RepoCandidate]:
    """Recursively bisect the ``created:LO..HI`` window until each shard fits the cap."""
    if lo > hi:
        return
    q = f"{base_query} created:{_iso(lo)}..{_iso(hi)}"
    total, _ = _search_page(
        q, 1, token=token, per_page=1, client=client, limiter=limiter
    )
    if total == 0:
        return
    if total <= _RESULT_CAP:
        for cand in search_repos(q, token=token, client=client, limiter=limiter):
            if cand.full_name not in seen:
                seen.add(cand.full_name)
                yield cand
        return
    if lo == hi:
        # A single day still overflows — split by stars instead of dates.
        logger.info("shard {} over cap on a single day; bisecting by stars", q)
        yield from _shard_by_stars(
            q, token=token, seen=seen, client=client, limiter=limiter
        )
        return
    mid = lo + (hi - lo) // 2
    logger.info(
        "shard {} has {} > {} results; bisecting {}..{}",
        q,
        total,
        _RESULT_CAP,
        _iso(lo),
        _iso(hi),
    )
    yield from _shard_by_dates(
        base_query, lo, mid, token=token, seen=seen, client=client, limiter=limiter
    )
    yield from _shard_by_dates(
        base_query,
        mid + timedelta(days=1),
        hi,
        token=token,
        seen=seen,
        client=client,
        limiter=limiter,
    )


def sharded_search(
    base_query: str,
    *,
    token: str | None,
    by: str = "created",
    start: date | None = None,
    end: date | None = None,
    client: httpx.Client | None = None,
    limiter: RateLimiter | None = None,
) -> Iterator[RepoCandidate]:
    """Enumerate a search population that exceeds the 1000-result cap.

    Attaches a range qualifier to ``base_query`` and recursively bisects it until every
    leaf shard reports ``total_count <= 1000``, then pages each leaf. ``by="created"``
    (default) bisects the creation-date window ``start..end`` (``2011-01-01..today`` by
    default); ``by="stars"`` bisects the stargazer range instead. Results are
    de-duplicated by ``full_name`` across shards.
    """
    client, limiter, owns = _managed(client, limiter)
    seen: set[str] = set()
    try:
        if by == "stars":
            yield from _shard_by_stars(
                base_query, token=token, seen=seen, client=client, limiter=limiter
            )
        else:
            lo = start or _EPOCH
            hi = end or datetime.now(UTC).date()
            yield from _shard_by_dates(
                base_query, lo, hi, token=token, seen=seen, client=client, limiter=limiter
            )
    finally:
        if owns:
            client.close()


def _seed_full_names() -> set[str]:
    """Return the lowercased ``owner/name`` of every repo already in the seed file."""
    try:
        doc = yaml.safe_load(_SEEDS_PATH.read_text(encoding="utf-8")) or {}
    except OSError:
        return set()
    names: set[str] = set()
    for section in _SEED_SECTIONS:
        for entry in doc.get(section, []) or []:
            if isinstance(entry, dict) and entry.get("repo"):
                names.add(str(entry["repo"]).lower())
    return names


def discover_all(
    *,
    token: str | None,
    queries: Iterable[str] | None = None,
    max_repos: int | None = None,
    client: httpx.Client | None = None,
    limiter: RateLimiter | None = None,
) -> dict[str, RepoCandidate]:
    """Run the curated query set and return a ``{full_name: RepoCandidate}`` map.

    Every query is fanned out through :func:`sharded_search`, results are merged and
    de-duplicated by ``full_name`` across all queries, and repos already present in
    ``seeds/official_repos.yaml`` are excluded (they are mirrored anyway). When a repo
    reappears with a more specific :attr:`~RepoCandidate.kind`, the classification is
    upgraded from ``"unknown"``. ``max_repos`` caps the number of *new* repos kept.
    """
    client, limiter, owns = _managed(client, limiter)
    query_list = list(queries) if queries is not None else list(DEFAULT_QUERIES)
    excluded = _seed_full_names()
    found: dict[str, RepoCandidate] = {}
    try:
        for query in query_list:
            before = len(found)
            for cand in sharded_search(
                query, token=token, client=client, limiter=limiter
            ):
                if cand.full_name.lower() in excluded:
                    continue
                existing = found.get(cand.full_name)
                if existing is None:
                    if max_repos is not None and len(found) >= max_repos:
                        logger.info("reached max_repos={}, stopping discovery", max_repos)
                        return found
                    found[cand.full_name] = cand
                elif existing.kind == "unknown" and cand.kind != "unknown":
                    found[cand.full_name] = cand
            logger.info(
                "query {!r}: +{} new repos ({} total)",
                query,
                len(found) - before,
                len(found),
            )
    finally:
        if owns:
            client.close()
    return found


def write_discovered(candidates: Iterable[RepoCandidate], path: Path = _DEFAULT_OUT) -> Path:
    """Persist candidates to a JSONL file (one :class:`RepoCandidate` per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for cand in candidates:
            fh.write(cand.model_dump_json())
            fh.write("\n")
    return path


def load_discovered(path: Path = _DEFAULT_OUT) -> list[RepoCandidate]:
    """Load candidates previously written by :func:`write_discovered`."""
    path = Path(path)
    out: list[RepoCandidate] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(RepoCandidate.model_validate_json(line))
    return out


def _to_dicts(mapping: dict[str, RepoCandidate]) -> list[dict]:
    """Convenience: a JSON-safe list of candidate dicts (e.g. for ``json.dump``)."""
    return [json.loads(c.model_dump_json()) for c in mapping.values()]
