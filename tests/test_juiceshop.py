"""Offline tests for the OWASP Juice Shop connector.

Fully deterministic: the HTTP GET for ``challenges.yml`` is mocked with
pytest-httpx and served from a committed sample fixture, so no network is
touched. We assert the connector's field mapping and that the normalized
``Challenge`` comes out redistributable with a mapped category.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctfhoard.connectors.juiceshop import CONNECTOR, JuiceShopConnector
from ctfhoard.normalize import normalize
from ctfhoard.schema import Category, Origin

_SAMPLE_URL = "https://example.test/challenges.yml"
_FIXTURE = Path(__file__).parent / "fixtures" / "juiceshop_sample.yml"


@pytest.fixture
def connector(tmp_path: Path, httpx_mock) -> JuiceShopConnector:
    httpx_mock.add_response(url=_SAMPLE_URL, text=_FIXTURE.read_text(encoding="utf-8"))
    return JuiceShopConnector(tmp_path / "work", challenges_url=_SAMPLE_URL)


def test_registry_binding() -> None:
    assert CONNECTOR is JuiceShopConnector
    assert JuiceShopConnector.cli_name == "juiceshop"
    assert JuiceShopConnector.origin is Origin.JUICESHOP


def test_discover_yields_all_challenges(connector: JuiceShopConnector) -> None:
    raws = list(connector.discover())
    assert len(raws) == 3
    assert [r.title for r in raws] == ["Password Hash Leak", "API-only XSS", "Admin Section"]
    assert all(r.origin is Origin.JUICESHOP for r in raws)
    assert all(r.event_name == "OWASP Juice Shop" for r in raws)


def test_field_mapping(connector: JuiceShopConnector) -> None:
    xss = next(r for r in connector.discover() if r.title == "API-only XSS")
    assert xss.raw_category == "XSS"
    assert xss.difficulty == "3"  # coerced from int to str
    assert xss.tags == ["Danger Zone"]
    assert "persisted XSS" in xss.description
    assert xss.extra["key"] == "restfulXssChallenge"

    # mitigationUrl becomes a reference-only writeup.
    assert len(xss.writeups) == 1
    wu = xss.writeups[0]
    assert wu.origin is Origin.JUICESHOP
    assert wu.is_inline is False
    assert str(wu.url).startswith("https://cheatsheetseries.owasp.org/")


def test_missing_optional_fields(connector: JuiceShopConnector) -> None:
    # 'Admin Section' has no mitigationUrl -> no writeups, but still a valid record.
    admin = next(r for r in connector.discover() if r.title == "Admin Section")
    assert admin.writeups == []
    assert admin.tags == ["Good for Demos"]
    assert admin.source is not None


def test_source_and_license(connector: JuiceShopConnector) -> None:
    raw = next(iter(connector.discover()))
    src = raw.source
    assert src is not None
    assert src.repo == "juice-shop/juice-shop"
    assert src.is_official is True
    assert src.license.spdx_id == "MIT"
    assert src.license.redistributable is True


def test_normalize_produces_valid_redistributable_challenge(
    connector: JuiceShopConnector,
) -> None:
    by_title = {r.title: normalize(r) for r in connector.discover()}

    # Juice Shop's OWASP-style labels ("XSS", "Broken Access Control", ...) are
    # web-application vulnerability classes, so the normalizer maps them to WEB
    # while preserving the raw label verbatim for traceability.
    xss = by_title["API-only XSS"]
    assert xss.category is Category.WEB
    assert xss.raw_category == "XSS"
    assert xss.difficulty == "3"
    assert len(xss.writeups) == 1

    for chal in by_title.values():
        assert chal.title
        assert chal.redistributable is True
        assert chal.license.spdx_id == "MIT"
        assert chal.event_name == "OWASP Juice Shop"
        assert isinstance(chal.category, Category)
