"""Offline tests for the GitHub *discovery* module.

No real network and no token: the GitHub search API is fully mocked with
``pytest-httpx`` via a callback that synthesizes ``total_count`` + ``items`` from the
request's ``q``/``page``/``per_page`` parameters. We prove the four behaviours that
matter:

* a query whose ``total_count`` exceeds the 1000-result cap triggers recursive
  ``created:`` date bisection into strictly smaller sub-windows (:func:`sharded_search`);
* results are de-duplicated by ``full_name`` across several queries, and seed repos are
  excluded (:func:`discover_all`);
* :attr:`RepoCandidate.kind` is inferred from topics/name/description (:func:`infer_kind`);
* candidates round-trip through ``write_discovered`` / ``load_discovered``.

The ``gh auth token`` path is exercised without spawning ``gh`` by monkeypatching env
and ``subprocess.run``.
"""

from __future__ import annotations

import contextlib
import re
from datetime import date, timedelta
from pathlib import Path

import httpx

from ctfhoard import discover
from ctfhoard.discover import (
    RepoCandidate,
    discover_all,
    infer_kind,
    load_discovered,
    resolve_token,
    sharded_search,
    write_discovered,
)
from ctfhoard.ratelimit import RateLimiter

_CREATED_RE = re.compile(r"created:(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})")


def _fast_limiter() -> RateLimiter:
    """A limiter with no delay so mocked tests never sleep.

    A huge refill rate + burst keeps the token bucket effectively unlimited, so the
    many requests emitted by the sharding tests never block on token replenishment.
    """
    return RateLimiter(rate=1e9, per=1.0, burst=1e9, min_interval=0.0)


def _repo_obj(full_name: str, *, topics: list[str] | None = None) -> dict:
    """A minimal search-API repository object."""
    return {
        "full_name": full_name,
        "html_url": f"https://github.com/{full_name}",
        "stargazers_count": 7,
        "pushed_at": "2024-01-02T03:04:05Z",
        "created_at": "2020-06-01T00:00:00Z",
        "topics": topics or ["ctf"],
        "description": "a ctf challenges repo",
        "license": {"spdx_id": "MIT"},
        "default_branch": "main",
        "size": 42,
    }


def _json(total: int, items: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"total_count": total, "items": items})


def _page_slice(names: list[str], page: int, per_page: int) -> list[dict]:
    start = (page - 1) * per_page
    return [_repo_obj(n) for n in names[start : start + per_page]]


@contextlib.contextmanager
def _captured_warnings():
    """Collect the module's loguru WARNING+ messages emitted inside the block."""
    records: list[str] = []
    sink_id = discover.logger.add(
        lambda msg: records.append(str(msg)), level="WARNING", format="{message}"
    )
    try:
        yield records
    finally:
        discover.logger.remove(sink_id)


# ---------------------------------------------------------------------------
# (a) bisection when total_count > 1000
# ---------------------------------------------------------------------------
def test_over_cap_triggers_date_bisection(httpx_mock) -> None:
    """A window that reports >1000 results is bisected into smaller date windows.

    The mock pretends exactly one repo was created per day in the window, so
    ``total_count`` == window length in days: windows wider than the 1000 cap must be
    split, windows within it are paged as leaves.
    """

    def callback(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        q = params["q"]
        page = int(params.get("page", "1"))
        per_page = int(params.get("per_page", "100"))
        m = _CREATED_RE.search(q)
        assert m, f"every sharded query must carry a created window: {q}"
        lo, hi = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        days = (hi - lo).days + 1
        if days > discover._RESULT_CAP:
            # Non-leaf: only the total matters, drives the bisection decision.
            return _json(days, [])
        names = [f"acme/ctf-{lo.isoformat()}-{i}" for i in range(days)]
        return _json(days, _page_slice(names, page, per_page))

    httpx_mock.add_callback(
        callback, url=re.compile(r"https://api\.github\.com/search/.*"), is_reusable=True
    )

    # 2020-01-01..2023-06-30 ≈ 1276 days > 1000 → must bisect into two ~638-day leaves.
    cands = list(
        sharded_search(
            "topic:ctf",
            token=None,
            start=date(2020, 1, 1),
            end=date(2023, 6, 30),
            client=httpx.Client(),
            limiter=_fast_limiter(),
        )
    )

    # Exceeding the single-query cap proves the shards' union beat the 1000 limit.
    assert len(cands) > discover._RESULT_CAP

    windows = set()
    for req in httpx_mock.get_requests():
        m = _CREATED_RE.search(req.url.params["q"])
        if m:
            windows.add((date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))))

    root_lo, root_hi = date(2020, 1, 1), date(2023, 6, 30)
    root = (root_lo, root_hi)
    # Root window plus at least the two bisected children were all queried.
    assert root in windows
    assert len(windows) >= 3

    # Real narrowing/containment: every non-root child window lies strictly inside the
    # root range AND spans strictly fewer days than the root (no re-querying the root,
    # no widening).
    root_span = root_hi - root_lo
    for lo_d, hi_d in windows:
        assert root_lo <= lo_d <= hi_d <= root_hi
        if (lo_d, hi_d) != root:
            assert (hi_d - lo_d) < root_span

    # The two first-level children partition the root at mid / mid+1 exactly — no gap,
    # no overlap — matching the bisection in ``_shard_by_dates``.
    mid = root_lo + (root_hi - root_lo) // 2
    assert (root_lo, mid) in windows
    assert (mid + timedelta(days=1), root_hi) in windows


# ---------------------------------------------------------------------------
# (a2) stars-bisection fallback for a single-day window over the cap
# ---------------------------------------------------------------------------
_STARS_RE = re.compile(r"stars:(\d+)\.\.(\d+)")


def test_single_day_overflow_falls_back_to_stars(httpx_mock, monkeypatch) -> None:
    """A single ``created:D..D`` day reporting >1000 forces ``stars:`` bisection.

    The mock models 1500 repos on one day, each with a distinct star count 0..1499, so
    only ``stars:`` sub-windows narrow enough (<=1000 span) become pageable leaves. We
    assert the fallback fires (queries gain ``stars:`` qualifiers), every repo is
    enumerated exactly once (deduped), and the ``lo>=hi`` recursion terminates without
    an off-by-one infinite loop.
    """
    monkeypatch.setattr(discover.time, "sleep", lambda _s: None)
    n_repos = 1500  # star value == index, so repo i matches stars:lo..hi iff lo<=i<=hi

    def callback(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        q = params["q"]
        page = int(params.get("page", "1"))
        per_page = int(params.get("per_page", "100"))
        sm = _STARS_RE.search(q)
        if sm is None:
            # Bare created:D..D probe — the whole day overflows the cap.
            return _json(n_repos, [])
        lo, hi = int(sm.group(1)), int(sm.group(2))
        matched = [i for i in range(n_repos) if lo <= i <= hi]
        if per_page == 1:
            return _json(len(matched), [])  # cheap total-only probe
        names = [f"acme/star-{i}" for i in matched]
        return _json(len(matched), _page_slice(names, page, per_page))

    httpx_mock.add_callback(
        callback, url=re.compile(r"https://api\.github\.com/search/.*"), is_reusable=True
    )

    cands = list(
        sharded_search(
            "topic:ctf",
            token=None,
            start=date(2020, 1, 1),
            end=date(2020, 1, 1),
            client=httpx.Client(),
            limiter=_fast_limiter(),
        )
    )

    # Fallback fired: some queries carried a stars: qualifier.
    star_queries = [
        req for req in httpx_mock.get_requests() if _STARS_RE.search(req.url.params["q"])
    ]
    assert star_queries, "expected stars: bisection to be triggered"

    # Every repo enumerated once (deduped across overlapping leaves). Each pageable leaf
    # spans <=1000 stars, so all 1500 are reachable across the star shards.
    names = [c.full_name for c in cands]
    assert len(names) == len(set(names)) == n_repos


# ---------------------------------------------------------------------------
# (a3) un-splittable star bucket still over the cap → observable warning
# ---------------------------------------------------------------------------
def test_unsplittable_star_bucket_warns(httpx_mock, monkeypatch) -> None:
    """A single star value holding >1000 repos can't be split further: warn, don't loop.

    All repos share the exact star count 5, so ``stars:`` bisection converges on the
    degenerate ``stars:5..5`` window that still reports >1000. The base case must emit a
    WARNING about the dropped overflow and terminate (no infinite recursion).
    """
    monkeypatch.setattr(discover.time, "sleep", lambda _s: None)
    total = 1200
    star_value = 5

    def callback(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        q = params["q"]
        page = int(params.get("page", "1"))
        per_page = int(params.get("per_page", "100"))
        sm = _STARS_RE.search(q)
        if sm is None:
            return _json(total, [])
        lo, hi = int(sm.group(1)), int(sm.group(2))
        if not (lo <= star_value <= hi):
            return _json(0, [])
        if per_page == 1:
            return _json(total, [])  # cheap total-only probe
        names = [f"acme/pop-{i}" for i in range(total)]
        return _json(total, _page_slice(names, page, per_page))

    httpx_mock.add_callback(
        callback, url=re.compile(r"https://api\.github\.com/search/.*"), is_reusable=True
    )

    with _captured_warnings() as warnings:
        cands = list(
            sharded_search(
                "topic:ctf",
                token=None,
                start=date(2020, 1, 1),
                end=date(2020, 1, 1),
                client=httpx.Client(),
                limiter=_fast_limiter(),
            )
        )

    # Reaching here at all proves the lo>=hi base case terminated (no infinite loop).
    assert any("un-splittable shard" in w for w in warnings)
    assert any(str(total - discover._RESULT_CAP) in w for w in warnings)
    # We still page the first 1000 of the overflowing bucket rather than nothing.
    assert len(cands) == discover._RESULT_CAP


# ---------------------------------------------------------------------------
# (a4) transient 5xx / network errors are retried, not fatal
# ---------------------------------------------------------------------------
def test_transient_5xx_page_retries_then_succeeds(httpx_mock, monkeypatch) -> None:
    """A leaf-shard page that returns 503 once then 200 is retried and succeeds."""
    monkeypatch.setattr(discover.time, "sleep", lambda _s: None)
    state = {"data_hits": 0}

    def callback(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        page = int(params.get("page", "1"))
        per_page = int(params.get("per_page", "100"))
        names = ["team/a", "team/b", "team/c"]
        if per_page == 100 and page == 1:
            state["data_hits"] += 1
            if state["data_hits"] == 1:
                return httpx.Response(503)  # transient server error on first try
        return _json(len(names), _page_slice(names, page, per_page))

    httpx_mock.add_callback(
        callback, url=re.compile(r"https://api\.github\.com/search/.*"), is_reusable=True
    )

    cands = list(
        sharded_search(
            "topic:ctf",
            token=None,
            start=date(2020, 1, 1),
            end=date(2020, 1, 1),
            client=httpx.Client(),
            limiter=_fast_limiter(),
        )
    )

    assert {c.full_name for c in cands} == {"team/a", "team/b", "team/c"}
    assert state["data_hits"] >= 2, "the 503 page must have been retried"


def test_persistent_failure_skips_shard_not_run(httpx_mock, monkeypatch) -> None:
    """A page failing persistently (503 / network) skips that shard, not the whole run.

    One query keeps returning 503 (retries exhausted → page skipped); another keeps
    raising a network error; a healthy query still yields its repos. The run must
    complete and preserve the repos it could enumerate.
    """
    monkeypatch.setattr(discover.time, "sleep", lambda _s: None)

    def callback(request: httpx.Request) -> httpx.Response:
        q = request.url.params["q"]
        page = int(request.url.params.get("page", "1"))
        per_page = int(request.url.params.get("per_page", "100"))
        if "good" in q:
            names = ["team/g1", "team/g2"]
            return _json(len(names), _page_slice(names, page, per_page))
        if "netfail" in q:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(503)  # "srvfail" query: persistent server error

    httpx_mock.add_callback(
        callback, url=re.compile(r"https://api\.github\.com/search/.*"), is_reusable=True
    )

    found = discover_all(
        token=None,
        queries=["good in:name", "srvfail in:name", "netfail in:name"],
        client=httpx.Client(),
        limiter=_fast_limiter(),
    )

    # The failing queries were skipped; the healthy query's repos survived.
    assert set(found) == {"team/g1", "team/g2"}


# ---------------------------------------------------------------------------
# (b) dedup across queries + seed exclusion
# ---------------------------------------------------------------------------
def test_discover_all_dedups_and_excludes_seeds(httpx_mock) -> None:
    """The same repo seen under two queries is kept once; seed repos are dropped."""

    def callback(request: httpx.Request) -> httpx.Response:
        q = request.url.params["q"]
        page = int(request.url.params.get("page", "1"))
        per_page = int(request.url.params.get("per_page", "100"))
        if "alpha" in q:
            # Includes a real seed repo, which must be excluded from discovery.
            names = ["team/a", "team/shared", "google/google-ctf"]
        elif "beta" in q:
            names = ["team/shared", "team/c"]
        else:
            names = []
        return _json(len(names), _page_slice(names, page, per_page))

    httpx_mock.add_callback(
        callback, url=re.compile(r"https://api\.github\.com/search/.*"), is_reusable=True
    )

    found = discover_all(
        token=None,
        queries=["alpha in:name", "beta in:name"],
        client=httpx.Client(),
        limiter=_fast_limiter(),
    )

    # team/shared appears in both queries → collapsed to one; google/google-ctf is a
    # curated seed → excluded entirely.
    assert set(found) == {"team/a", "team/shared", "team/c"}
    assert "google/google-ctf" not in found


def test_discover_all_respects_max_repos(httpx_mock) -> None:
    def callback(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        per_page = int(request.url.params.get("per_page", "100"))
        names = [f"team/repo-{i}" for i in range(5)]
        return _json(len(names), _page_slice(names, page, per_page))

    httpx_mock.add_callback(
        callback, url=re.compile(r"https://api\.github\.com/search/.*"), is_reusable=True
    )

    found = discover_all(
        token=None,
        queries=["ctf in:name"],
        max_repos=3,
        client=httpx.Client(),
        limiter=_fast_limiter(),
    )
    assert len(found) == 3


# ---------------------------------------------------------------------------
# (c) kind inference
# ---------------------------------------------------------------------------
def test_infer_kind_from_topics_name_description() -> None:
    assert infer_kind("x/ctf-writeups", None, []) == "writeups"
    assert infer_kind("x/repo", "our CTF write-ups", []) == "writeups"
    assert infer_kind("x/repo", None, ["ctf", "writeup"]) == "writeups"
    assert infer_kind("x/ctf-challenges", None, ["ctf-challenges"]) == "sources"
    assert infer_kind("x/repo", "challenge sources", []) == "sources"
    assert infer_kind("x/random", "just a tool", ["security"]) == "unknown"
    # Writeup signal wins over a co-occurring source signal.
    assert infer_kind("x/ctf-challenges-writeups", None, []) == "writeups"


def test_repo_candidate_from_api_maps_fields() -> None:
    cand = RepoCandidate.from_api(_repo_obj("owner/ctf-writeups", topics=["ctf", "writeup"]))
    assert cand.full_name == "owner/ctf-writeups"
    assert cand.stars == 7
    assert cand.license_spdx == "MIT"
    assert cand.default_branch == "main"
    assert cand.size_kb == 42
    assert cand.kind == "writeups"
    assert cand.created_at is not None and cand.created_at.year == 2020


def test_repo_candidate_noassertion_license_becomes_none() -> None:
    obj = _repo_obj("o/n")
    obj["license"] = {"spdx_id": "NOASSERTION"}
    assert RepoCandidate.from_api(obj).license_spdx is None
    obj["license"] = None
    assert RepoCandidate.from_api(obj).license_spdx is None


# ---------------------------------------------------------------------------
# (d) write / load round-trip
# ---------------------------------------------------------------------------
def test_write_load_roundtrip(tmp_path: Path) -> None:
    cands = [
        RepoCandidate.from_api(_repo_obj("a/b", topics=["ctf-challenges"])),
        RepoCandidate.from_api(_repo_obj("c/d-writeups", topics=["writeup"])),
    ]
    path = tmp_path / "discovered_repos.jsonl"
    written = write_discovered(cands, path)
    assert written == path

    loaded = load_discovered(path)
    assert [c.full_name for c in loaded] == ["a/b", "c/d-writeups"]
    assert loaded[0].kind == "sources"
    assert loaded[1].kind == "writeups"
    assert loaded == cands


# ---------------------------------------------------------------------------
# token resolution (no real `gh` invocation)
# ---------------------------------------------------------------------------
def test_resolve_token_prefers_explicit_then_env_then_gh(monkeypatch) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert resolve_token("explicit-tok") == "explicit-tok"

    monkeypatch.setenv("GH_TOKEN", "env-tok")
    assert resolve_token(None) == "env-tok"
    monkeypatch.delenv("GH_TOKEN", raising=False)

    class _Proc:
        stdout = "gh-cli-tok\n"

    monkeypatch.setattr(discover.subprocess, "run", lambda *a, **k: _Proc())
    assert resolve_token(None) == "gh-cli-tok"


def test_git_repo_connector_ingests_discovered(tmp_path: Path) -> None:
    """The git_repo connector appends discovered repos to its seed walk list."""
    from ctfhoard.connectors.git_repo import GitRepoConnector

    disc = tmp_path / "discovered_repos.jsonl"
    write_discovered(
        [
            RepoCandidate.from_api(_repo_obj("new/ctf-repo", topics=["ctf-challenges"])),
            # A repo already in the seeds must not be added twice.
            RepoCandidate.from_api(_repo_obj("google/google-ctf")),
        ],
        disc,
    )
    conn = GitRepoConnector(tmp_path / "work", discovered_path=disc)
    repos = [s["repo"] for s in conn.seeds]
    assert "new/ctf-repo" in repos
    assert repos.count("google/google-ctf") == 1
    new_seed = next(s for s in conn.seeds if s["repo"] == "new/ctf-repo")
    assert new_seed["kind"] == "sources"
    assert new_seed["license"] is None
    assert new_seed["official"] is False
