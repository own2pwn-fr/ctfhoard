"""Offline tests for the Hackropole connector.

Fully deterministic: the sitemap, the challenge page, and the file download are
all mocked with pytest-httpx and served from committed fixtures — no network is
touched. We assert the connector's HTML field extraction (category, tags,
difficulty, description, author, writeups), that it materializes the challenge's
files under ``local_dir``, and that ``normalize()`` turns the result into a
redistributable ``Challenge`` with a mapped category and detected source.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ctfhoard.connectors.hackropole import CONNECTOR, HackropoleConnector
from ctfhoard.normalize import normalize
from ctfhoard.schema import Category, Origin

_FIXTURES = Path(__file__).parent / "fixtures"
_SITEMAP_URL = "https://hackropole.fr/fr/sitemap.xml"
_CRYPTO_URL = "https://hackropole.fr/fr/challenges/crypto/fcsc2026-crypto-a-une-vache-pres/"
_WEB_URL = "https://hackropole.fr/fr/challenges/web/fcsc2023-web-salty-authentication/"
_FILE_URL = (
    "https://hackropole.fr/challenges/fcsc2026-crypto-a-une-vache-pres/public/a-une-vache-pres.py"
)
_FILE_BODY = b"# SageMath solution stub\nprint('FCSC{...}')\n"


def _mock_sitemap(httpx_mock) -> None:
    httpx_mock.add_response(
        url=_SITEMAP_URL,
        text=(_FIXTURES / "hackropole_sitemap.xml").read_text(encoding="utf-8"),
    )


def _mock_crypto_page(httpx_mock) -> None:
    httpx_mock.add_response(
        url=_CRYPTO_URL,
        text=(_FIXTURES / "hackropole_challenge.html").read_text(encoding="utf-8"),
    )
    httpx_mock.add_response(url=_FILE_URL, content=_FILE_BODY)


def test_registry_binding() -> None:
    assert CONNECTOR is HackropoleConnector
    assert HackropoleConnector.cli_name == "hackropole"
    assert HackropoleConnector.origin is Origin.HACKROPOLE


def test_sitemap_enumeration_respects_limit(tmp_path: Path, httpx_mock) -> None:
    # Only the sitemap and the first challenge (+ its file) are fetched with limit=1.
    _mock_sitemap(httpx_mock)
    _mock_crypto_page(httpx_mock)
    conn = HackropoleConnector(tmp_path / "work", limit=1)
    raws = list(conn.discover())
    assert len(raws) == 1
    assert raws[0].extra["slug"] == "fcsc2026-crypto-a-une-vache-pres"


@pytest.fixture
def raw(tmp_path: Path, httpx_mock):
    _mock_sitemap(httpx_mock)
    _mock_crypto_page(httpx_mock)
    conn = HackropoleConnector(tmp_path / "work", limit=1)
    return next(iter(conn.discover()))


def test_field_extraction(raw) -> None:
    assert raw.origin is Origin.HACKROPOLE
    assert raw.title == "À une vache près"
    assert raw.event_name == "FCSC 2026"
    assert raw.year == 2026
    assert raw.raw_category == "crypto"  # taken from the URL path segment
    assert raw.tags == ["sagemath"]  # category + edition badges are excluded
    assert raw.difficulty == "3/5"  # three filled stars
    assert "courbe elliptique" in raw.description
    assert raw.extra["authors"] == ["jp"]


def test_writeups_parsed_and_deduped(raw) -> None:
    # The solutions table links the same writeup twice (two stretched links per row);
    # they must collapse to a single Writeup.
    assert len(raw.writeups) == 1
    wu = raw.writeups[0]
    assert wu.origin is Origin.HACKROPOLE
    assert str(wu.url).endswith("/cecf5439-cd60-4ac3-895f-2cb4e055ab4b/")


def test_files_downloaded_into_local_dir(raw) -> None:
    assert raw.local_dir is not None
    local = Path(raw.local_dir)
    downloaded = local / "a-une-vache-pres.py"
    assert downloaded.exists()
    assert downloaded.read_bytes() == _FILE_BODY


def test_source_is_official_etalab(raw) -> None:
    src = raw.source
    assert src is not None
    assert src.is_official is True
    assert str(src.url) == _CRYPTO_URL
    assert src.license.spdx_id == "etalab-2.0"
    assert src.license.redistributable is True


def test_normalize_produces_redistributable_challenge(raw) -> None:
    chal = normalize(raw)
    assert chal.category is Category.CRYPTO  # "crypto" mapped
    assert chal.raw_category == "crypto"
    assert chal.event_name == "FCSC 2026"
    assert chal.difficulty == "3/5"
    # The .py handout is challenge source -> whitebox flag set, license carries over.
    assert chal.has_source is True
    assert "python" in chal.solve_languages
    assert chal.redistributable is True
    assert chal.license.spdx_id == "etalab-2.0"
    assert any(str(f.path) == "a-une-vache-pres.py" for f in chal.files)


def test_enumerates_every_challenge_without_limit(tmp_path: Path, httpx_mock) -> None:
    # No limit: both challenge URLs in the sitemap are visited. We serve the same
    # (crypto) page fixture for both — the count and per-URL slug/category are what
    # matter here.
    _mock_sitemap(httpx_mock)
    page = (_FIXTURES / "hackropole_challenge.html").read_text(encoding="utf-8")
    httpx_mock.add_response(url=_CRYPTO_URL, text=page)
    httpx_mock.add_response(url=_WEB_URL, text=page)
    # Both pages reference the same handout URL; each visited challenge downloads it
    # into its own dir, so the single-use mock is registered once per fetch.
    httpx_mock.add_response(url=_FILE_URL, content=_FILE_BODY)
    httpx_mock.add_response(url=_FILE_URL, content=_FILE_BODY)
    conn = HackropoleConnector(tmp_path / "work")
    raws = list(conn.discover())
    assert len(raws) == 2
    slugs = {r.extra["slug"] for r in raws}
    assert slugs == {
        "fcsc2026-crypto-a-une-vache-pres",
        "fcsc2023-web-salty-authentication",
    }
    # Category is read from each URL independently even though the HTML is identical.
    by_slug = {r.extra["slug"]: r for r in raws}
    assert by_slug["fcsc2023-web-salty-authentication"].raw_category == "web"


def test_one_bad_page_does_not_abort_crawl(tmp_path: Path, monkeypatch) -> None:
    # Regression: a single challenge page that persistently 5xxes (the retrying
    # PoliteClient finally re-raises an HTTPStatusError) must not sink the whole
    # crawl — the challenges before and after it must still be yielded.
    url1 = "https://hackropole.fr/fr/challenges/crypto/fcsc2026-crypto-a-une-vache-pres/"
    url2 = "https://hackropole.fr/fr/challenges/pwn/fcsc2024-pwn-boom/"
    url3 = "https://hackropole.fr/fr/challenges/web/fcsc2023-web-salty-authentication/"
    page = (_FIXTURES / "hackropole_challenge.html").read_text(encoding="utf-8")

    conn = HackropoleConnector(tmp_path / "work")
    monkeypatch.setattr(conn, "_challenge_urls", lambda: [url1, url2, url3])
    # This test is about crawl resilience, not file mirroring — skip downloads.
    monkeypatch.setattr(conn, "_download_files", lambda *a, **k: None)

    def fake_get(url: str, **kwargs) -> httpx.Response:
        request = httpx.Request("GET", url)
        if url == url2:
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("persistent 500", request=request, response=response)
        return httpx.Response(200, text=page, request=request)

    monkeypatch.setattr(conn._client, "get", fake_get)

    raws = list(conn.discover())
    # URLs 1 and 3 survive; the broken middle page is skipped, not fatal.
    assert [r.extra["slug"] for r in raws] == [
        "fcsc2026-crypto-a-une-vache-pres",
        "fcsc2023-web-salty-authentication",
    ]


def test_truncated_download_is_not_trusted(tmp_path: Path, httpx_mock) -> None:
    # Regression: a transfer that delivers fewer bytes than its declared
    # Content-Length (an aborted/killed download) must NOT be promoted to a trusted
    # final handout — it must be discarded (only a .part could remain, and here even
    # that is cleaned up) so the next run re-fetches it.
    httpx_mock.add_response(url=_FILE_URL, content=b"partial", headers={"Content-Length": "999"})
    conn = HackropoleConnector(tmp_path / "work")
    dest = tmp_path / "work" / "chall"
    dest.mkdir(parents=True)
    name = "a-une-vache-pres.py"

    conn._download_files([(_FILE_URL, name)], dest)

    assert not (dest / name).exists()  # truncated body never trusted as final
    assert not (dest / f"{name}.part").exists()  # scratch file cleaned up


def test_leftover_part_is_redownloaded(tmp_path: Path, httpx_mock) -> None:
    # Regression: a leftover ``<name>.part`` from a SIGKILLed run is an incomplete
    # transfer; it must be discarded and the file re-downloaded (then atomically
    # promoted to its final name), never mistaken for a finished handout.
    httpx_mock.add_response(url=_FILE_URL, content=_FILE_BODY)
    conn = HackropoleConnector(tmp_path / "work")
    dest = tmp_path / "work" / "chall"
    dest.mkdir(parents=True)
    name = "a-une-vache-pres.py"
    (dest / f"{name}.part").write_bytes(b"trunc")  # truncated leftover from a killed run

    conn._download_files([(_FILE_URL, name)], dest)

    final = dest / name
    assert final.exists()
    assert final.read_bytes() == _FILE_BODY  # fully re-downloaded
    assert not (dest / f"{name}.part").exists()  # leftover cleaned up / promoted
