"""CTFtime connector: the metadata graph + a polite writeup delta-crawl.

CTFtime (https://ctftime.org) is the canonical index of CTF events. It exposes two
very different surfaces, and this connector bridges them:

* a clean **JSON API** (``/api/v1/``) for event metadata — fast, no crawl-delay;
* **HTML-only** writeup pages that must be crawled *politely* (robots.txt declares
  ``Crawl-delay: 10`` ≈ one request / 10 s, plus a ``Content-signal:
  ai-train=no, use=reference`` — so writeups are kept as **reference**: we hold the
  link, and only capture inline text when it is hosted on the ctftime page itself).

There is **no writeups endpoint** in the JSON API; the writeup graph only exists as
HTML:

* per-event task list  ``/event/{event_id}/tasks/`` — an HTML table, one row per
  task (``/task/{task_id}``) carrying Points, Tags and a writeups count;
* per-task page        ``/task/{task_id}``          — a table of writeups, each row
  ``/writeup/{writeup_id}`` with a rating and the author team;
* individual writeup   ``/writeup/{writeup_id}``    — INCONSISTENT: older ones embed
  the writeup body inline (``div.well#id_description``); many modern ones are just an
  external link ("Original writeup" / "Official writeup") to a blog or GitHub. We
  parse the inline body when present, else extract that external URL.

``/task/tags/`` is Disallowed by robots.txt — we never fetch tag pages.

Each emitted :class:`~ctfhoard.schema.RawChallenge` is one *task*, carrying its event
context and its writeup references. These records generally have **no source files**
(``has_source=False``): CTFtime holds writeups and metadata, not challenge source —
that is expected. This connector is the metadata graph + delta crawl; for the actual
challenge sources and unrolled writeup *text* there are existing mirrors we do not
reinvent — ``sajjadium/ctf-archives`` (challenge sources) and the Hugging Face
dataset ``justinwangx/CTFtime-unrolled`` (writeup text).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from selectolax.parser import HTMLParser, Node

from ctfhoard.connectors.base import Connector
from ctfhoard.http import PoliteClient, make_client
from ctfhoard.normalize import map_category
from ctfhoard.ratelimit import RateLimiter
from ctfhoard.schema import (
    Category,
    CtfEvent,
    LicenseInfo,
    Origin,
    RawChallenge,
    Source,
    Writeup,
)

#: JSON API root (fast, no crawl-delay).
API_BASE = "https://ctftime.org/api/v1"

#: HTML site root (crawled politely, one request / 10 s).
SITE_BASE = "https://ctftime.org"

#: CTFtime's declared crawl-delay (robots.txt). The default for real runs; tests
#: override it to 0 so the offline suite does not sleep.
CRAWL_DELAY_SECONDS = 10.0

#: Matches "/task/123", "/writeup/123", "/team/123" hrefs → the trailing id.
_ID_RE = re.compile(r"/(?:task|writeup|team)/(\d+)")

#: Writeup wells that are a bare external reference start with this phrase.
_EXTERNAL_MARKERS = ("original writeup", "official writeup")


class CTFtimeConnector(Connector):
    """Emit one ``RawChallenge`` per CTFtime task, with event + writeup context."""

    cli_name = "ctftime"
    origin = Origin.CTFTIME

    def __init__(
        self,
        workdir: Path,
        start: int | None = None,
        finish: int | None = None,
        event_limit: int = 100,
        max_events: int | None = None,
        max_writeups_per_task: int | None = None,
        *,
        html_min_interval: float = CRAWL_DELAY_SECONDS,
    ) -> None:
        """Bound and cap the crawl.

        ``start``/``finish`` are unix timestamps bounding which events to pull
        (default: the last ~2 years, overridable). ``event_limit`` is the JSON API
        page size (the API has no offset — we slide the ``start``/``finish`` time
        window). ``max_events`` / ``max_writeups_per_task`` cap work for
        incremental runs and testing. ``html_min_interval`` is the crawl-delay
        enforced on the HTML crawl; keep the 10 s default for real runs, set it to
        0 in tests so they do not hang.
        """
        super().__init__(workdir)
        now = int(datetime.now(UTC).timestamp())
        two_years = int(timedelta(days=730).total_seconds())
        self.start = start if start is not None else now - two_years
        self.finish = finish if finish is not None else now
        self.event_limit = event_limit
        self.max_events = max_events
        self.max_writeups_per_task = max_writeups_per_task

        # Two HTTP layers: a plain client for the fast JSON API, and a polite
        # (rate-limited, retrying) wrapper for the HTML crawl. The limiter's hard
        # ``min_interval`` is what honors the 10 s crawl-delay; the generous token
        # bucket never adds delay of its own (and vanishes entirely at interval 0).
        self._api = make_client()
        self._html_limiter = RateLimiter(
            rate=1000.0, per=1.0, burst=1000, min_interval=html_min_interval
        )
        self._html = PoliteClient(make_client(), self._html_limiter)

    # -- JSON API: event metadata ------------------------------------------

    def _fetch_events(self) -> Iterator[dict]:
        """Yield raw event dicts across the time window.

        The API has no offset parameter, so we paginate by sliding the ``start``
        of the ``[start, finish]`` window forward past the last event we saw. Event
        ids are de-duplicated across window boundaries. Stops at ``max_events``.
        """
        seen: set[int] = set()
        window_start = self.start
        emitted = 0
        while window_start < self.finish:
            resp = self._api.get(
                f"{API_BASE}/events/",
                params={
                    "limit": self.event_limit,
                    "start": window_start,
                    "finish": self.finish,
                },
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                return
            max_start = window_start
            new_in_batch = 0
            for event in batch:
                eid = event.get("id")
                if eid in seen:
                    continue
                seen.add(eid)
                new_in_batch += 1
                yield event
                emitted += 1
                if self.max_events is not None and emitted >= self.max_events:
                    return
                ts = _iso_to_ts(event.get("start"))
                if ts is not None:
                    max_start = max(max_start, ts)
            # Full page → there may be more; slide the window to (not past) the last
            # event's start. Sliding to ``max_start`` — instead of ``max_start + 1`` —
            # keeps events that share that boundary second but did not fit on this
            # page: they reappear on the next page and the ``seen`` set dedupes the
            # ones already emitted. A short page (or a page that added nothing new —
            # e.g. every event shares one second) means we are done, so the window
            # never advancing can never loop forever.
            if len(batch) < self.event_limit or new_in_batch == 0:
                return
            window_start = max_start

    @staticmethod
    def _build_event(event: dict) -> CtfEvent:
        """Map a raw API event dict onto a :class:`CtfEvent`."""
        start = _iso_to_dt(event.get("start"))
        return CtfEvent(
            ctftime_event_id=event.get("id"),
            ctftime_series_id=event.get("ctf_id"),
            name=event.get("title") or "unknown-event",
            year=start.year if start else None,
            start=start,
            finish=_iso_to_dt(event.get("finish")),
            weight=event.get("weight"),
            format=event.get("format"),
            ctftime_url=event.get("ctftime_url") or None,
            homepage=event.get("url") or None,
        )

    # -- HTML crawl: tasks & writeups --------------------------------------

    def _fetch_html(self, url: str) -> HTMLParser | None:
        """Politely GET ``url`` and parse it, or return None on a non-200.

        Pages come and go; a missing one must degrade gracefully (skip) rather than
        sink the whole crawl.
        """
        try:
            resp = self._html.get(url)
        except Exception:  # noqa: BLE001 — one bad page must not abort the crawl
            return None
        if resp.status_code != 200:
            return None
        return HTMLParser(resp.text)

    def _parse_tasks(self, tree: HTMLParser) -> list[dict]:
        """Parse an event's ``/tasks/`` table into task descriptors.

        Each row's first cell links the task (``/task/{id}``); the second cell holds
        the points and any ``span.label`` tags. The writeups-count cell also links
        ``/task/{id}`` — we take the *first* task link per row (the name) to avoid it.
        """
        tasks: dict[int, dict] = {}
        for row in tree.css("table.table-striped tr"):
            link = row.css_first('a[href^="/task/"]')
            if link is None:
                continue  # header row / non-task row
            tid = _extract_id(link.attributes.get("href"))
            if tid is None or tid in tasks:
                continue
            cells = row.css("td")
            points = _parse_int(cells[1].text(strip=True)) if len(cells) > 1 else None
            tags = [s.text(strip=True) for s in row.css("span.label") if s.text(strip=True)]
            tasks[tid] = {
                "id": tid,
                "name": link.text(strip=True) or f"task-{tid}",
                "points": points,
                "tags": tags,
            }
        return list(tasks.values())

    def _parse_task_writeups(self, tree: HTMLParser) -> list[dict]:
        """Parse a task page's writeups table into ``{id, rating, author}`` rows."""
        out: list[dict] = []
        seen: set[int] = set()
        for row in tree.css("tr"):
            link = row.css_first('a[href^="/writeup/"]')
            if link is None:
                continue
            wid = _extract_id(link.attributes.get("href"))
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            cells = row.css("td")
            rating = _parse_float(cells[1].text(strip=True)) if len(cells) > 1 else None
            team = row.css_first('a[href^="/team/"]')
            out.append(
                {
                    "id": wid,
                    "rating": rating,
                    "author": team.text(strip=True) if team else None,
                }
            )
        return out

    def _build_writeup(self, wid: int, meta: dict) -> Writeup | None:
        """Fetch and classify one writeup: inline body vs external reference link.

        Honors the ``ai-train=no, use=reference`` signal: inline text is only
        captured when it is hosted on the ctftime page itself (``is_inline=True``);
        otherwise we keep just the external reference URL (``is_inline=False``).
        """
        tree = self._fetch_html(f"{SITE_BASE}/writeup/{wid}")
        if tree is None:
            return None
        retrieved = datetime.now(UTC)

        body = _inline_body(tree)
        if body:
            return Writeup(
                url=f"{SITE_BASE}/writeup/{wid}",
                origin=Origin.CTFTIME,
                author=meta.get("author"),
                text=body,
                is_inline=True,
                rating=meta.get("rating"),
                retrieved_at=retrieved,
            )

        external = _external_url(tree)
        if external:
            origin = Origin.GITHUB if "github.com" in external.lower() else Origin.OTHER
            return Writeup(
                url=external,
                origin=origin,
                author=meta.get("author"),
                is_inline=False,
                rating=meta.get("rating"),
                retrieved_at=retrieved,
            )
        # Neither an inline body nor an external link — keep the ctftime page as the
        # bare reference so the edge in the graph is not lost.
        return Writeup(
            url=f"{SITE_BASE}/writeup/{wid}",
            origin=Origin.CTFTIME,
            author=meta.get("author"),
            is_inline=False,
            rating=meta.get("rating"),
            retrieved_at=retrieved,
        )

    def _writeups_for_task(self, task_id: int) -> list[Writeup]:
        tree = self._fetch_html(f"{SITE_BASE}/task/{task_id}")
        if tree is None:
            return []
        rows = self._parse_task_writeups(tree)
        if self.max_writeups_per_task is not None:
            rows = rows[: self.max_writeups_per_task]
        writeups: list[Writeup] = []
        for row in rows:
            wu = self._build_writeup(row["id"], row)
            if wu is not None:
                writeups.append(wu)
        return writeups

    # -- driver -------------------------------------------------------------

    def discover(self) -> Iterator[RawChallenge]:
        for raw_event in self._fetch_events():
            event = self._build_event(raw_event)
            tasks_tree = self._fetch_html(f"{SITE_BASE}/event/{event.ctftime_event_id}/tasks/")
            if tasks_tree is None:
                continue  # no task list (private/undocumented event) — skip
            for task in self._parse_tasks(tasks_tree):
                writeups = self._writeups_for_task(task["id"])
                tags = task["tags"]
                yield RawChallenge(
                    origin=Origin.CTFTIME,
                    title=task["name"],
                    event_name=event.name,
                    year=event.year,
                    raw_category=_pick_raw_category(tags),
                    tags=tags,
                    points=task["points"],
                    writeups=writeups,
                    source=Source(
                        origin=Origin.CTFTIME,
                        url=f"{SITE_BASE}/task/{task['id']}",
                        is_official=False,
                        # Writeups/metadata are author copyright and reference-only —
                        # NOT redistributable by default (conservative LicenseInfo()).
                        license=LicenseInfo(),
                    ),
                    extra={
                        "ctftime_event_id": event.ctftime_event_id,
                        "ctftime_series_id": event.ctftime_series_id,
                        "ctftime_task_id": task["id"],
                    },
                )


# -- module-level parsing helpers ------------------------------------------


def _extract_id(href: str | None) -> int | None:
    if not href:
        return None
    m = _ID_RE.search(href)
    return int(m.group(1)) if m else None


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group(0)) if m else None


def _parse_float(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None


def _iso_to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _iso_to_ts(value: str | None) -> int | None:
    dt = _iso_to_dt(value)
    return int(dt.timestamp()) if dt else None


def _inline_body(tree: HTMLParser) -> str | None:
    """The writeup body when hosted inline on the ctftime page, else None."""
    node = tree.css_first("div.well#id_description")
    if node is None:
        return None
    text = node.text(strip=True)
    return text or None


def _external_url(tree: HTMLParser) -> str | None:
    """The 'Original/Official writeup' external reference URL, if any."""
    for well in tree.css("div.well"):
        if well.attributes.get("id") == "id_description":
            continue
        if not any(marker in well.text().lower() for marker in _EXTERNAL_MARKERS):
            continue
        link: Node | None = well.css_first("a[href]")
        if link is None:
            continue
        href = link.attributes.get("href")
        if href and href.lower().startswith("http"):
            return href
    return None


def _pick_raw_category(tags: list[str]) -> str | None:
    """Derive a raw category label from a task's tags.

    Prefer the first tag that maps to a known :class:`Category` (so
    ``normalize()`` classifies it correctly); fall back to the first tag; None when
    there are no tags.
    """
    for tag in tags:
        if map_category(tag) is not Category.UNKNOWN:
            return tag
    return tags[0] if tags else None


CONNECTOR = CTFtimeConnector
