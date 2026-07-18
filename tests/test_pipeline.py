"""Offline tests for the disk-bounded streaming mirror pipeline.

No real network and no real cloning: :meth:`GitRepoConnector.discover` is monkeypatched
to yield fabricated :class:`RawChallenge` records backed by a real temp file tree (so the
corpus materializer copies real bytes), and :func:`ctfhoard.hf.publish_corpus` /
:func:`~ctfhoard.hf.publish_catalog` are replaced with recorders that never touch the
network. The tests assert the streaming contract: materialize → catalog shard → publish →
cleanup, plus resume, failure isolation, and the disk guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctfhoard import hf, pipeline
from ctfhoard.connectors.git_repo import GitRepoConnector
from ctfhoard.schema import Origin, RawChallenge, Source

_SHA = "0123456789abcdef0123456789abcdef01234567"


def _make_raw(tmp_path: Path, repo: str, title: str) -> RawChallenge:
    """Fabricate one RawChallenge whose ``local_dir`` holds real source files."""
    local = tmp_path / "clones" / repo.replace("/", "__") / title
    local.mkdir(parents=True, exist_ok=True)
    (local / "chal.py").write_text("print('pwn')\n", encoding="utf-8")
    (local / "Dockerfile").write_text("FROM ubuntu:22.04\n", encoding="utf-8")
    return RawChallenge(
        origin=Origin.GITHUB,
        title=title,
        event_name="SomeCTF",
        year=2024,
        raw_category="pwn",
        local_dir=str(local),
        source=Source(origin=Origin.GITHUB, repo=repo, commit_sha=_SHA),
    )


class _Recorder:
    """Records publish calls without any network access."""

    def __init__(self) -> None:
        self.corpus_calls: list[dict] = []
        self.catalog_calls: list[dict] = []

    def publish_corpus(self, corpus_dir, **kwargs) -> str:
        self.corpus_calls.append({"corpus_dir": Path(corpus_dir), **kwargs})
        return "recorded://corpus"

    def publish_catalog(self, catalog_dir, **kwargs) -> str:
        self.catalog_calls.append({"catalog_dir": Path(catalog_dir), **kwargs})
        return "recorded://catalog"


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Patch the HF publishers on the ``hf`` module used by the pipeline."""
    rec = _Recorder()
    monkeypatch.setattr(hf, "publish_corpus", rec.publish_corpus)
    monkeypatch.setattr(hf, "publish_catalog", rec.publish_catalog)
    return rec


def _patch_discover(monkeypatch: pytest.MonkeyPatch, raws: list[RawChallenge]) -> None:
    """Make every GitRepoConnector yield ``raws`` from discover() (no cloning)."""
    monkeypatch.setattr(GitRepoConnector, "discover", lambda self: iter(raws))


def test_mirror_repo_materializes_publishes_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    raw = _make_raw(tmp_path, "acme/pwn", "target_practice")
    _patch_discover(monkeypatch, [raw])

    result = pipeline.mirror_repo(
        "acme/pwn",
        data_dir=data_dir,
        dataset="acme/ds",
        publish=True,
        keep_local=False,
        max_repo_size_mb=4096,
        token="t",
    )

    # Outcome + provenance.
    assert result.status == "ok"
    assert result.repo == "acme/pwn"
    assert result.commit_sha == _SHA
    assert result.n_challenges == 1
    assert result.n_files >= 2  # chal.py + Dockerfile copied in
    assert result.bytes > 0

    # Publish was invoked against this repo's isolated corpus subtree.
    assert len(recorder.corpus_calls) == 1
    assert recorder.corpus_calls[0]["path_in_repo"] == "corpus"
    assert recorder.corpus_calls[0]["dataset"] == "acme/ds"
    assert len(recorder.catalog_calls) == 1

    # Catalog shard KEPT; raw clone + staging corpus REMOVED.
    slug = "acme__pwn"
    shard = data_dir / "catalog" / slug / "challenges.jsonl"
    assert shard.exists()
    assert not (data_dir / "raw" / slug).exists()
    assert not (data_dir / "staging" / slug).exists()

    # The kept shard carries the HF-layout corpus_path.
    records = [json.loads(line) for line in shard.read_text().splitlines() if line]
    assert records and records[0]["corpus_path"].startswith("corpus/")


def test_mirror_repo_keep_local_retains_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    _patch_discover(monkeypatch, [_make_raw(tmp_path, "acme/keep", "chal")])

    result = pipeline.mirror_repo(
        "acme/keep",
        data_dir=data_dir,
        dataset="acme/ds",
        publish=False,
        keep_local=True,
        max_repo_size_mb=4096,
        token=None,
    )

    assert result.status == "ok"
    slug = "acme__keep"
    # Staging kept (keep_local=True), raw clone still always removed.
    assert (data_dir / "staging" / slug / "corpus").exists()
    assert not (data_dir / "raw" / slug).exists()
    # publish=False → no publish calls.
    assert recorder.corpus_calls == []
    assert recorder.catalog_calls == []


def test_mirror_all_resume_skips_done_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Pre-seed the manifest: repo A already 'ok'.
    manifest = data_dir / "mirror_state.jsonl"
    manifest.write_text(
        json.dumps({"repo": "acme/A", "status": "ok", "commit_sha": _SHA}) + "\n",
        encoding="utf-8",
    )

    # A is skipped by resume, so discover() is only ever reached for B.
    b_raw = _make_raw(tmp_path, "acme/B", "bchal")
    monkeypatch.setattr(GitRepoConnector, "discover", lambda self: iter([b_raw]))

    results = pipeline.mirror_all(
        ["acme/A", "acme/B"],
        data_dir=data_dir,
        dataset="acme/ds",
        publish=True,
        keep_local=False,
        max_repo_size_mb=4096,
        token="t",
        resume=True,
    )

    # A skipped (not re-emitted), B processed.
    assert [r.repo for r in results] == ["acme/B"]
    assert results[0].status == "ok"

    # Manifest now records both A (pre-seeded) and B (appended).
    repos_in_manifest = [
        json.loads(line)["repo"] for line in manifest.read_text().splitlines() if line
    ]
    assert repos_in_manifest == ["acme/A", "acme/B"]


def test_mirror_all_failed_repo_does_not_abort_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    good_raw = _make_raw(tmp_path, "acme/good", "gchal")

    # ``workdir`` is shared (data/raw) for both repos, so branch on ``only`` instead.
    def _discover_by_only(self):
        if "acme/bad" in (self.only or set()):
            raise RuntimeError("clone timed out")
        return iter([good_raw])

    monkeypatch.setattr(GitRepoConnector, "discover", _discover_by_only)

    results = pipeline.mirror_all(
        ["acme/bad", "acme/good"],
        data_dir=data_dir,
        dataset="acme/ds",
        publish=False,
        keep_local=False,
        max_repo_size_mb=4096,
        token=None,
        resume=False,
    )

    assert [r.status for r in results] == ["failed", "ok"]
    assert results[0].error and "clone timed out" in results[0].error
    # The failing repo left no raw clone behind (cleanup ran in finally).
    assert not (data_dir / "raw" / "acme__bad").exists()


def test_mirror_repo_disk_guard_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    from collections import namedtuple

    data_dir = tmp_path / "data"
    _patch_discover(monkeypatch, [_make_raw(tmp_path, "acme/big", "chal")])

    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        pipeline.shutil, "disk_usage", lambda _p: usage(100, 99, 1 * 1024**3)
    )

    result = pipeline.mirror_repo(
        "acme/big",
        data_dir=data_dir,
        dataset="acme/ds",
        publish=True,
        keep_local=False,
        max_repo_size_mb=4096,
        token="t",
    )

    assert result.status == "skipped"
    assert result.error and "low disk" in result.error
    # Guard fired before any clone/publish.
    assert recorder.corpus_calls == []
    assert not (data_dir / "raw" / "acme__big").exists()


def test_resolve_repo_list_sources(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    # only=... overrides everything and de-dups preserving order.
    assert pipeline.resolve_repo_list(
        data_dir=data_dir, source="seeds", only=["a/b", "a/b", "c/d"]
    ) == ["a/b", "c/d"]

    # seeds: read from a synthetic seeds YAML.
    seeds = tmp_path / "seeds.yaml"
    seeds.write_text(
        "official_sources:\n"
        "  - repo: org/one\n"
        "community_archives:\n"
        "  - repo: org/two\n",
        encoding="utf-8",
    )
    assert pipeline.resolve_repo_list(
        data_dir=data_dir, source="seeds", only=None, seeds_path=seeds
    ) == ["org/one", "org/two"]
