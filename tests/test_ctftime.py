"""Offline tests for the CTFtime connector.

Fully deterministic: the JSON events API, the per-event task list, the per-task
writeup table, and the individual writeup pages are all mocked with pytest-httpx
from committed fixtures — no network is touched. The 10 s crawl-delay is disabled
via ``html_min_interval=0`` so the suite does not hang (the default stays 10 s for
real runs).

We assert the events -> tasks -> writeups traversal, that inline writeups keep their
body (``is_inline=True``, ``origin=CTFTIME``) while external ones keep only the
reference link (``is_inline=False``, ``origin=GITHUB`` for github.com), that event
context (name/year/series id) is populated, and that ``normalize()`` yields a valid
``Challenge`` that is source-less and not redistributable by default.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from ctfhoard.connectors.ctftime import (
    CONNECTOR,
    CRAWL_DELAY_SECONDS,
    CTFtimeConnector,
)
from ctfhoard.normalize import normalize
from ctfhoard.schema import Category, Origin

_FIXTURES = Path(__file__).parent / "fixtures"

# Deterministic window so the (regex-matched) API request is stable.
_START = 1_640_995_200  # 2022-01-01
_FINISH = 1_711_929_600  # 2024-04-01

_API_RE = re.compile(r"^https://ctftime\.org/api/v1/events/")


def _fx(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _mock_events(httpx_mock) -> None:
    httpx_mock.add_response(url=_API_RE, json=json.loads(_fx("ctftime_events.json")))


def _mock_tasks(httpx_mock) -> None:
    httpx_mock.add_response(
        url="https://ctftime.org/event/1656/tasks/", text=_fx("ctftime_tasks.html")
    )
    httpx_mock.add_response(
        url="https://ctftime.org/event/2100/tasks/", text=_fx("ctftime_tasks_empty.html")
    )
    httpx_mock.add_response(url="https://ctftime.org/task/23295", text=_fx("ctftime_task.html"))
    httpx_mock.add_response(
        url="https://ctftime.org/task/23296", text=_fx("ctftime_task_empty.html")
    )


def _mock_writeups(httpx_mock, *, external: bool = True) -> None:
    httpx_mock.add_response(
        url="https://ctftime.org/writeup/34834", text=_fx("ctftime_writeup_inline.html")
    )
    if external:
        httpx_mock.add_response(
            url="https://ctftime.org/writeup/39000", text=_fx("ctftime_writeup_external.html")
        )


def _connector(tmp_path: Path, **kwargs) -> CTFtimeConnector:
    return CTFtimeConnector(
        tmp_path / "work",
        start=_START,
        finish=_FINISH,
        html_min_interval=0.0,
        **kwargs,
    )


def test_registry_binding() -> None:
    assert CONNECTOR is CTFtimeConnector
    assert CTFtimeConnector.cli_name == "ctftime"
    assert CTFtimeConnector.origin is Origin.CTFTIME
    # The real crawl-delay default is preserved for production runs.
    assert CRAWL_DELAY_SECONDS == 10.0


@pytest.fixture
def raws(tmp_path: Path, httpx_mock):
    _mock_events(httpx_mock)
    _mock_tasks(httpx_mock)
    _mock_writeups(httpx_mock)
    return list(_connector(tmp_path).discover())


def test_traversal_yields_one_record_per_task(raws) -> None:
    # Event 1656 has two tasks (zigzag, whatsapp); event 2100 has an empty task list
    # and must be skipped gracefully.
    assert {r.title for r in raws} == {"zigzag", "whatsapp"}


def test_event_and_task_context(raws) -> None:
    zig = next(r for r in raws if r.title == "zigzag")
    assert zig.origin is Origin.CTFTIME
    assert zig.event_name == "corCTF 2022"
    assert zig.year == 2022
    assert zig.tags == ["zig", "pwn", "heap"]
    assert zig.points == 225
    # raw_category is derived from the tags (first tag that maps to a Category).
    assert zig.raw_category == "pwn"
    assert zig.extra["ctftime_event_id"] == 1656
    assert zig.extra["ctftime_series_id"] == 1006  # stable series id
    assert zig.extra["ctftime_task_id"] == 23295


def test_inline_vs_external_writeups(raws) -> None:
    zig = next(r for r in raws if r.title == "zigzag")
    assert len(zig.writeups) == 2
    by_inline = {w.is_inline: w for w in zig.writeups}

    inline = by_inline[True]
    assert inline.origin is Origin.CTFTIME
    assert inline.text is not None and "GeneralPurposeAllocator" in inline.text
    assert str(inline.url) == "https://ctftime.org/writeup/34834"
    assert inline.author == "ret2school"
    assert inline.rating is None  # "not rated"

    external = by_inline[False]
    assert external.origin is Origin.GITHUB  # github.com link
    assert external.text is None  # reference only, no body captured
    assert str(external.url).startswith("https://github.com/ret2school/ctf")
    assert external.author == "7rocky"
    assert external.rating == 4.0


def test_task_without_writeups_still_yields(raws) -> None:
    whatsapp = next(r for r in raws if r.title == "whatsapp")
    assert whatsapp.writeups == []
    assert whatsapp.raw_category == "web"
    assert whatsapp.points == 145


def test_source_is_reference_only(raws) -> None:
    zig = next(r for r in raws if r.title == "zigzag")
    src = zig.source
    assert src is not None
    assert src.origin is Origin.CTFTIME
    assert str(src.url) == "https://ctftime.org/task/23295"
    assert src.is_official is False
    # Writeups are author copyright / reference-only: not redistributable by default.
    assert src.license.redistributable is False
    assert src.license.spdx_id is None


def test_normalize_produces_valid_sourceless_challenge(raws) -> None:
    zig = next(r for r in raws if r.title == "zigzag")
    chal = normalize(zig)
    assert chal.title == "zigzag"
    assert chal.category is Category.PWN  # "pwn" mapped
    assert chal.event_name == "corCTF 2022"
    assert chal.year == 2022
    assert chal.event is not None and chal.event.name == "corCTF 2022"
    # CTFtime holds writeups/metadata, not challenge source -> no source files.
    assert chal.has_source is False
    assert chal.redistributable is False
    assert len(chal.writeups) == 2


def test_max_writeups_per_task_caps_crawl(tmp_path: Path, httpx_mock) -> None:
    # With a cap of 1, only the first writeup row is fetched — the external one is
    # never requested (so it is intentionally not mocked).
    _mock_events(httpx_mock)
    _mock_tasks(httpx_mock)
    _mock_writeups(httpx_mock, external=False)
    raws = list(_connector(tmp_path, max_writeups_per_task=1).discover())
    zig = next(r for r in raws if r.title == "zigzag")
    assert len(zig.writeups) == 1
    assert zig.writeups[0].is_inline is True


def test_max_events_caps_events(tmp_path: Path, httpx_mock) -> None:
    # Only the first event is processed; the second event's task list is never hit.
    _mock_events(httpx_mock)
    httpx_mock.add_response(
        url="https://ctftime.org/event/1656/tasks/", text=_fx("ctftime_tasks.html")
    )
    httpx_mock.add_response(url="https://ctftime.org/task/23295", text=_fx("ctftime_task.html"))
    httpx_mock.add_response(
        url="https://ctftime.org/task/23296", text=_fx("ctftime_task_empty.html")
    )
    _mock_writeups(httpx_mock)
    raws = list(_connector(tmp_path, max_events=1).discover())
    assert {r.event_name for r in raws} == {"corCTF 2022"}


# -- window-sliding pagination ---------------------------------------------

# Three distinct start-seconds inside [_START, _FINISH]; T1 is the page boundary
# shared by two events (one that fits on the first page, one that does not).
_T0 = "2022-06-01T00:00:00+00:00"
_T1 = "2022-07-01T00:00:00+00:00"
_T2 = "2022-08-01T00:00:00+00:00"


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


def _window_api_callback(events: list[dict]):
    """A fake CTFtime events API honoring the ``start``/``finish``/``limit`` window.

    Mirrors the real endpoint: it returns the first ``limit`` events whose start is
    within ``[start, finish]``, sorted ascending by start. This is what makes the
    off-by-one window bug observable — a naive ``start = max_start + 1`` slide skips
    events sharing the boundary second, exactly as it would against the live API.
    """

    def callback(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        ws, fin, lim = int(params["start"]), int(params["finish"]), int(params["limit"])
        page = sorted(
            (e for e in events if ws <= _ts(e["start"]) <= fin),
            key=lambda e: _ts(e["start"]),
        )[:lim]
        return httpx.Response(200, json=page)

    return callback


def test_window_pagination_keeps_boundary_second_events(tmp_path: Path, httpx_mock) -> None:
    # Regression (#25): a FULL first page (len == event_limit) ends on second T1,
    # which a second event also shares but which did not fit on that page. The window
    # must slide to T1 (not T1 + 1) so the boundary event is re-served and kept; the
    # seen-set dedupes the ones already emitted. Every distinct event exactly once.
    events = [
        {"id": 9001, "ctf_id": 1, "title": "e1", "start": _T0, "finish": _T0},
        {"id": 9002, "ctf_id": 1, "title": "e2", "start": _T0, "finish": _T0},
        {"id": 9003, "ctf_id": 1, "title": "e3", "start": _T1, "finish": _T1},  # boundary (fits)
        {"id": 9004, "ctf_id": 1, "title": "e4", "start": _T1, "finish": _T1},  # boundary (cut)
        {"id": 9005, "ctf_id": 1, "title": "e5", "start": _T2, "finish": _T2},
    ]
    httpx_mock.add_callback(_window_api_callback(events), url=_API_RE, is_reusable=True)

    got = list(_connector(tmp_path, event_limit=3)._fetch_events())

    ids = [e["id"] for e in got]
    assert ids == [9001, 9002, 9003, 9004, 9005]  # boundary event 9004 not dropped
    assert len(ids) == len(set(ids))  # emitted exactly once (no duplicate)


def test_window_pagination_all_same_second_terminates(tmp_path: Path, httpx_mock) -> None:
    # More events than a page, all sharing one start-second: the window can never
    # advance, so termination must come from the seen-set + zero-new-events guard.
    # (If it did not, list() below would hang forever.)
    events = [
        {"id": 8000 + i, "ctf_id": 1, "title": f"e{i}", "start": _T1, "finish": _T1}
        for i in range(5)
    ]
    httpx_mock.add_callback(_window_api_callback(events), url=_API_RE, is_reusable=True)

    got = list(_connector(tmp_path, event_limit=3)._fetch_events())

    ids = [e["id"] for e in got]
    assert len(ids) == len(set(ids))  # no duplicate emission
    assert set(ids).issubset({8000, 8001, 8002, 8003, 8004})
