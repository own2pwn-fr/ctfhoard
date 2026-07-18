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
import subprocess
from pathlib import Path

import pytest

from ctfhoard import mirror
from ctfhoard.connectors import git_repo
from ctfhoard.connectors.git_repo import (
    CONNECTOR,
    GitRepoConnector,
    _origin_for_repo,
    walk_repo,
)
from ctfhoard.normalize import map_category, normalize
from ctfhoard.schema import Category, LicenseInfo, Origin

#: Canonical MIT opening line — enough for ``licenses.detect_from_text`` to fire.
_MIT_TEXT = (
    "MIT License\n\n"
    "Copyright (c) 2024 Example\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy "
    'of this software and associated documentation files (the "Software"), to deal '
    "in the Software without restriction.\n"
)

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


# --- [HIGH] repo ROOT is a candidate challenge leaf -------------------------


def test_walk_single_challenge_at_root(tmp_path: Path) -> None:
    """(Case A) A repo whose ROOT directly holds artifacts is ONE challenge, not zero.

    Before the fix ``_iter_challenge_dirs`` recursed straight into the root's sub-dirs
    and, with none present, silently emitted nothing — the whole challenge dropped.
    """
    root = tmp_path / "repo"
    _write(root / "Dockerfile", "FROM ubuntu:22.04")
    _write(root / "app.py", "print('pwn me')")

    raws = list(walk_repo(root, _seed(), "acme/single-chall", _SHA))

    assert len(raws) == 1
    r = raws[0]
    # Title derives from the repo name (relative path is empty at the root).
    assert r.title == "single-chall"
    assert r.local_dir == str(root)
    assert r.source.path_in_repo in (None, "", ".")


def test_walk_component_only_root_public_private(tmp_path: Path) -> None:
    """(Case B) A root whose sub-dirs are ALL components is ONE challenge, not two.

    Before the fix a ``public``/``private`` root emitted two bogus challenges named
    after its parts; now it collapses to a single leaf = the root.
    """
    root = tmp_path / "repo"
    _write(root / "public" / "Dockerfile", "FROM php:8")
    _write(root / "private" / "flag.py", "FLAG = 'x'")

    raws = list(walk_repo(root, _seed(), "acme/pub-priv-chall", _SHA))

    titles = {r.title for r in raws}
    assert titles == {"pub-priv-chall"}
    assert "public" not in titles
    assert "private" not in titles


def test_walk_component_only_root_src_deploy(tmp_path: Path) -> None:
    """(Case B, second layout) ``src``/``deploy`` root also collapses to one leaf."""
    root = tmp_path / "repo"
    _write(root / "src" / "main.c", "int main(){return 0;}")
    _write(root / "deploy" / "Dockerfile", "FROM ubuntu:22.04")

    raws = list(walk_repo(root, _seed(), "acme/src-deploy-chall", _SHA))

    titles = {r.title for r in raws}
    assert titles == {"src-deploy-chall"}
    assert "src" not in titles
    assert "deploy" not in titles


def test_root_leaf_fix_preserves_multichallenge_walk(tmp_path: Path) -> None:
    """The justCTF-style container walk still yields one leaf per challenge dir."""
    root = tmp_path / "repo"
    _build_fake_repo(root)

    raws = list(walk_repo(root, _seed(), "sajjadium/ctf-archives", _SHA))

    assert {r.title for r in raws} == {"Target Practice", "babyphp", "nested"}


def test_multichallenge_archive_root_does_not_collapse(tmp_path: Path) -> None:
    """A multi-challenge archive root fans out into its nested challenges, not one leaf.

    Regression for the cryptohack/ctf_archive collapse: a stray artifact at the archive
    ROOT (there a top-level ``docker_deploy.py``) made ``_has_direct_artifacts(root)``
    fire, so the whole repo was emitted as a SINGLE challenge titled after the repo with
    ``path_in_repo == '.'`` and every file swallowed. The root must now PREFER its many
    ``<event>/<chal>/…`` sub-challenges over collapsing.
    """
    root = tmp_path / "repo"
    # Dozens-style archive: several events, each with nested challenge dirs.
    for event, chals in {
        "EventA-2023": ("alpha", "beta"),
        "EventB-2024": ("gamma", "delta"),
        "EventC-2025": ("epsilon",),
    }.items():
        for chal in chals:
            _write(root / event / chal / "Dockerfile", "FROM ubuntu:22.04")
            _write(root / event / chal / f"{chal}.py", "print('x')")
    # A stray artifact + a README sitting at the very root (the collapse trigger).
    _write(root / "docker_deploy.py", "import sys")
    _write(root / "README.md", "# archive")

    raws = list(walk_repo(root, _seed(), "cryptohack/ctf_archive", _SHA))

    # One challenge per nested chal dir — never the collapsed root.
    assert len(raws) == 5
    titles = {r.title for r in raws}
    assert titles == {"alpha", "beta", "gamma", "delta", "epsilon"}
    # No leaf is the repo root: no empty/dot path_in_repo, none titled after the repo.
    assert all(r.source.path_in_repo not in (None, "", ".") for r in raws)
    assert "ctf_archive" not in titles
    paths = {r.source.path_in_repo for r in raws}
    assert "EventA-2023/alpha" in paths
    assert "EventC-2025/epsilon" in paths


# --- category derived from the challenge DIRECTORY NAME ----------------------


def test_category_derived_from_challenge_dir_name(tmp_path: Path) -> None:
    """Google-CTF encodes the category as a token in the dir NAME, not a clean segment.

    ``2017-finals-crypto-bender`` / ``pwn-heat`` / ``re-arcade`` / ``rev-polymorph``
    carry no ancestor path segment that maps to a category, so before the fix they all
    fell through to UNKNOWN. The name is now scanned for a WHOLE-token category keyword
    (``re``/``rev`` for reverse too), and the derived ``raw_category`` normalizes.
    """
    root = tmp_path / "repo"
    # No ancestor segment maps to a category -> the name is the only signal.
    _write(root / "2017" / "finals" / "2017-finals-crypto-bender" / "bender.py", "x = 1")
    _write(root / "2024" / "quals" / "pwn-heat" / "Dockerfile", "FROM ubuntu:22.04")
    _write(root / "2024" / "quals" / "re-arcade" / "arcade.c", "int main(){return 0;}")
    _write(root / "2021" / "quals" / "rev-polymorph" / "poly.py", "print(1)")
    # Whole-token guard: ``web`` must NOT match inside ``webhook``.
    _write(root / "2024" / "quals" / "webhook-abuse" / "note.py", "pass")

    raws = list(walk_repo(root, _seed(), "google/google-ctf", _SHA))
    by = {r.title: r for r in raws}

    bender = by["2017-finals-crypto-bender"]
    assert bender.raw_category is not None
    assert map_category(bender.raw_category) is Category.CRYPTO
    assert normalize(bender).category is Category.CRYPTO
    assert bender.year == 2017
    assert bender.edition == "finals"

    assert map_category(by["pwn-heat"].raw_category) is Category.PWN
    assert normalize(by["pwn-heat"]).category is Category.PWN
    # ``re`` and ``rev`` are both recognized as WHOLE tokens for reverse.
    assert normalize(by["re-arcade"]).category is Category.REVERSE
    assert normalize(by["rev-polymorph"]).category is Category.REVERSE

    # ``webhook`` is not a bare ``web`` token, so no false crypto/web hit.
    assert by["webhook-abuse"].raw_category is None
    assert normalize(by["webhook-abuse"]).category is Category.UNKNOWN


def test_kctf_component_wrapper_collapses_to_one_challenge(tmp_path: Path) -> None:
    """A kCTF challenge dir wrapping {attachments, challenge, healthcheck} is ONE leaf.

    The 2024 Google-CTF layout fragmented into bogus leaves literally titled
    ``attachments`` / ``challenge`` / ``healthcheck`` (their component sub-dirs). The
    presence of such a strong root-marker sub-dir now pins the PARENT as the single
    challenge, and no component sub-dir is ever emitted as a challenge on its own.
    """
    root = tmp_path / "repo"
    base = root / "2024" / "quals" / "web-sappy"
    _write(base / "attachments" / "handout.zip", "PK\x03\x04")
    _write(base / "challenge" / "Dockerfile", "FROM node:20")
    _write(base / "challenge" / "src" / "app.js", "console.log(1)")
    _write(base / "healthcheck" / "healthcheck.py", "print('ok')")
    # Challenge-level metadata sits at the parent top level (no direct artifact there).
    _write(base / "metadata.yaml", "name: sappy\n")

    raws = list(walk_repo(root, _seed(), "google/google-ctf", _SHA))

    titles = {r.title for r in raws}
    assert titles == {"web-sappy"}
    for component in ("attachments", "challenge", "healthcheck"):
        assert component not in titles
    assert len(raws) == 1
    # The single leaf still carries all its parts and derives its category from the name.
    assert raws[0].local_dir == str(base)
    assert map_category(raws[0].raw_category) is Category.WEB


# --- [MEDIUM] git subprocess timeout ----------------------------------------


def test_run_git_timeout_raises_mirror_error(monkeypatch) -> None:
    """A stalled git call must surface as ``MirrorError`` (not hang / not TimeoutExpired)."""

    def fake_run(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr(mirror.subprocess, "run", fake_run)

    with pytest.raises(mirror.MirrorError):
        mirror._run_git(["clone", "https://example.invalid/x", "/tmp/does-not-matter"])


# --- [MEDIUM] discovered-repo size guard ------------------------------------


def test_discovered_repo_over_size_cap_is_skipped(tmp_path: Path) -> None:
    """A discovered repo bigger than the cap is never added to the walk (not cloned)."""
    from ctfhoard.discover import RepoCandidate, write_discovered

    disc = tmp_path / "discovered.jsonl"
    write_discovered(
        [
            RepoCandidate(
                full_name="fake/small-repo",
                html_url="https://github.com/fake/small-repo",
                size_kb=100,
                kind="sources",
            ),
            RepoCandidate(
                full_name="fake/huge-repo",
                html_url="https://github.com/fake/huge-repo",
                size_kb=5000,
                kind="sources",
            ),
        ],
        path=disc,
    )

    conn = GitRepoConnector(tmp_path / "work", discovered_path=disc, max_repo_size_mb=1)
    repos = {s["repo"] for s in conn.seeds}

    assert "fake/small-repo" in repos
    assert "fake/huge-repo" not in repos


# --- [LOW] LICENSE filename variants ----------------------------------------


def test_detect_license_matches_filename_variant(tmp_path: Path) -> None:
    """A ``LICENSE-MIT`` (not the exact name ``LICENSE``) is still detected as MIT."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "LICENSE-MIT").write_text(_MIT_TEXT, encoding="utf-8")

    info = mirror.detect_repo_license(root, declared_spdx=None)

    assert info.spdx_id == "MIT"
    assert info.redistributable is True
    assert info.source_file == "LICENSE-MIT"


# --- [LOW] bounded reads of untrusted probe files ---------------------------


def test_read_text_bounded_reads_only_prefix(tmp_path: Path) -> None:
    """A huge probe file is read only up to the byte limit (never slurped whole)."""
    big = tmp_path / "LICENSE"
    big.write_text("HEAD" + "X" * 5_000_000, encoding="utf-8")

    text = mirror._read_text_bounded(big, limit=4)

    assert text == "HEAD"
    assert len(text) == 4


def test_oversized_challenge_json_is_skipped(tmp_path: Path, monkeypatch) -> None:
    """An over-cap ``challenge.json`` is treated as absent, not parsed into memory."""
    monkeypatch.setattr(git_repo, "_MAX_CHALLENGE_JSON_BYTES", 8)
    root = tmp_path / "repo"
    chal = root / "chal"
    _write(chal / "challenge.json", json.dumps({"name": "HugeName", "flag": "flag{x}"}))
    _write(chal / "app.py", "print(1)")

    raws = list(walk_repo(root, _seed(), "some/repo", _SHA))

    assert len(raws) == 1
    r = raws[0]
    # The oversized metadata file is skipped: title falls back to the dir name and no
    # flag/name is lifted from the (unparsed) challenge.json.
    assert r.title == "chal"
    assert r.flag is None
