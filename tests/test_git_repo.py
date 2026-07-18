"""Offline, deterministic tests for the generic ``git_repo`` connector.

No network and no cloning: we build a synthetic repo tree on ``tmp_path`` covering
the two dominant layouts (an NYU-style dir with ``challenge.json`` and plain
``Dockerfile``+source dirs, including a nested category and a component ``src/``
sub-dir) and drive :func:`walk_repo` directly over it. We assert the challenge-dir
heuristic (leaf selection, no parent/child double-emit, noise skipped), the
path→metadata derivation, provenance (commit SHA / sub-path), and that
:func:`normalize` turns the results into ``has_source`` challenges with a
content fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path

from ctfhoard.connectors.git_repo import (
    CONNECTOR,
    GitRepoConnector,
    _origin_for_repo,
    walk_repo,
)
from ctfhoard.normalize import normalize
from ctfhoard.schema import Category, LicenseInfo, Origin

_SHA = "0123456789abcdef0123456789abcdef01234567"


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_fake_repo(root: Path) -> None:
    """A synthetic clone mixing NYU (challenge.json) and sajjadium-style layouts."""
    # NYU-bench style: <year>/<event>/<category>/<challenge>/challenge.json
    nyu = root / "2023" / "CSAW-Quals" / "pwn" / "target_practice"
    _write(
        nyu / "challenge.json",
        json.dumps(
            {
                "name": "Target Practice",
                "category": "pwn",
                "flag": "flag{deadbeef}",
                "description": "Warm up your ROP skills.",
            }
        ),
    )
    _write(nyu / "target_practice.c", "int main(){return 0;}")
    _write(nyu / "Dockerfile", "FROM ubuntu:22.04")

    # sajjadium style: CTFs/<event>/<year>/Quals/<category>/<challenge>
    base = root / "CTFs" / "SomeCTF" / "2022" / "Quals" / "web"
    _write(base / "babyphp" / "Dockerfile", "FROM php:8")
    _write(base / "babyphp" / "index.php", "<?php echo 1;")
    _write(base / "babyphp" / "README.md", "# babyphp")

    # A second challenge in the SAME category dir (container must recurse) whose
    # source lives partly in a component src/ sub-dir (must NOT be emitted twice).
    _write(base / "nested" / "Dockerfile", "FROM python:3.12")
    _write(base / "nested" / "app.py", "print('hi')")
    _write(base / "nested" / "src" / "helper.py", "def h(): pass")

    # Noise: asset-only dir and a fake .git dir must yield nothing.
    _write(root / "assets" / "logo.png", "PNG")
    _write(root / ".git" / "config", "[core]")


def _seed(**over) -> dict:
    seed = {"repo": "Nautilus-Institute/quals-2023", "kind": "sources", "official": True}
    seed.update(over)
    return seed


def test_registry_binding() -> None:
    assert CONNECTOR is GitRepoConnector
    assert GitRepoConnector.cli_name == "git_repo"
    assert GitRepoConnector.origin is Origin.GITHUB


def test_origin_resolution() -> None:
    assert _origin_for_repo("google/google-ctf") is Origin.GOOGLE_CTF
    assert _origin_for_repo("sajjadium/ctf-archives") is Origin.SAJJADIUM
    assert _origin_for_repo("pwncollege/ctf-archive") is Origin.PWNCOLLEGE
    assert _origin_for_repo("NYU-LLM-CTF/NYU_CTF_Bench") is Origin.NYU_CTF_BENCH
    assert _origin_for_repo("justcatthefish/justctf-2023") is Origin.GITHUB


def test_walk_emits_one_per_challenge_leaf(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _build_fake_repo(root)
    lic = LicenseInfo(spdx_id="MIT", redistributable=True, confidence=1.0)

    raws = list(walk_repo(root, _seed(), "Nautilus-Institute/quals-2023", _SHA, license_info=lic))

    # Exactly three leaves: the two Dockerfile+source dirs and the challenge.json dir.
    # No parent container and no component src/ dir are emitted.
    assert len(raws) == 3
    assert {r.title for r in raws} == {"Target Practice", "babyphp", "nested"}
    # All provenance carries the pinned commit and the correct repo sub-path.
    for r in raws:
        assert r.source is not None
        assert r.source.commit_sha == _SHA
        assert r.source.repo == "Nautilus-Institute/quals-2023"
        assert r.source.is_official is True
        assert r.local_dir is not None
        assert r.source.path_in_repo == Path(r.local_dir).relative_to(root).as_posix()


def test_challenge_json_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _build_fake_repo(root)
    raws = list(walk_repo(root, _seed(official=True), "NYU-LLM-CTF/NYU_CTF_Bench", _SHA))
    target = next(r for r in raws if r.title == "Target Practice")

    assert target.raw_category == "pwn"
    assert target.year == 2023
    assert target.event_name == "CSAW-Quals"
    assert target.flag == "flag{deadbeef}"
    assert target.description == "Warm up your ROP skills."
    assert target.origin is Origin.NYU_CTF_BENCH
    assert target.local_dir.endswith("2023/CSAW-Quals/pwn/target_practice")


def test_path_derivation_sajjadium_layout(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _build_fake_repo(root)
    raws = list(walk_repo(root, _seed(), "sajjadium/ctf-archives", _SHA))
    baby = next(r for r in raws if r.title == "babyphp")

    assert baby.event_name == "SomeCTF"  # the "CTFs" wrapper is not the event
    assert baby.year == 2022
    assert baby.edition == "Quals"
    assert baby.raw_category == "web"
    assert baby.origin is Origin.SAJJADIUM


def test_license_falls_back_to_declared_spdx(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _build_fake_repo(root)
    # No license_info passed and no LICENSE file -> detect from the seed's SPDX.
    raws = list(walk_repo(root, _seed(license="MIT"), "some/repo", _SHA))
    assert raws
    assert all(r.source.license.spdx_id == "MIT" for r in raws)
    assert all(r.source.license.redistributable for r in raws)


def test_normalize_yields_sourced_fingerprinted_challenges(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _build_fake_repo(root)
    lic = LicenseInfo(spdx_id="MIT", redistributable=True, confidence=1.0)
    raws = list(walk_repo(root, _seed(), "Nautilus-Institute/quals-2023", _SHA, license_info=lic))

    by_title = {r.title: normalize(r) for r in raws}
    for chal in by_title.values():
        assert chal.has_source is True
        assert chal.content_fingerprint is not None
        assert chal.sources and chal.sources[0].commit_sha == _SHA

    assert by_title["babyphp"].category is Category.WEB
    assert by_title["Target Practice"].category is Category.PWN
    # The Dockerfile ships with each challenge -> docker flag set by normalize().
    assert by_title["nested"].docker is True


def test_connector_loads_and_filters_real_seeds(tmp_path: Path) -> None:
    # Uses the committed seeds file; no cloning happens until discover() is called.
    conn = GitRepoConnector(tmp_path / "work", only=["google/google-ctf"])
    assert len(conn.seeds) == 1
    assert conn.seeds[0]["repo"] == "google/google-ctf"
    assert conn.seeds[0]["license"] == "Apache-2.0"
