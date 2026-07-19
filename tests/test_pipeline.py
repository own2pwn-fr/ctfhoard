"""Offline tests for the disk-bounded, commit-batched mirror pipeline.

No real network and no real cloning: :meth:`GitRepoConnector.discover` is monkeypatched
to yield fabricated :class:`RawChallenge` records backed by a real temp file tree (so the
corpus materializer copies real bytes), and :func:`ctfhoard.hf.publish_corpus` /
:func:`~ctfhoard.hf.publish_catalog` are replaced with recorders that never touch the
network. The tests assert the batching contract: stage many repos into ONE shared batch,
publish the whole batch in a single corpus + catalog commit, and mark repos ``ok`` ONLY
after that publish succeeds — plus resume, failure isolation, and the disk guard.
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
    """Records publish calls without any network access.

    ``raise_on_corpus_call`` (1-based) makes the Nth ``publish_corpus`` raise, to
    exercise the batch-publish-failure path. Each catalog call snapshots the number of
    shard sub-directories present at flush time, which equals the batch's repo count.
    """

    def __init__(self, raise_on_corpus_call: int | None = None) -> None:
        self.corpus_calls: list[dict] = []
        self.catalog_calls: list[dict] = []
        self._raise_on = raise_on_corpus_call

    def publish_corpus(self, corpus_dir, **kwargs) -> str:
        self.corpus_calls.append({"corpus_dir": Path(corpus_dir), **kwargs})
        if self._raise_on is not None and len(self.corpus_calls) == self._raise_on:
            raise RuntimeError("HF 429: rate limited")
        return "recorded://corpus"

    def publish_catalog(self, catalog_dir, **kwargs) -> str:
        cdir = Path(catalog_dir)
        n_shards = sum(1 for p in cdir.glob("*/challenges.jsonl")) if cdir.exists() else 0
        self.catalog_calls.append({"catalog_dir": cdir, "n_shards": n_shards, **kwargs})
        return "recorded://catalog"


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Patch the HF publishers on the ``hf`` module used by the pipeline."""
    rec = _Recorder()
    monkeypatch.setattr(hf, "publish_corpus", rec.publish_corpus)
    monkeypatch.setattr(hf, "publish_catalog", rec.publish_catalog)
    return rec


def _install_recorder(monkeypatch: pytest.MonkeyPatch, rec: _Recorder) -> None:
    """Wire a specific recorder onto the ``hf`` publishers."""
    monkeypatch.setattr(hf, "publish_corpus", rec.publish_corpus)
    monkeypatch.setattr(hf, "publish_catalog", rec.publish_catalog)


def _patch_discover(monkeypatch: pytest.MonkeyPatch, raws: list[RawChallenge]) -> None:
    """Make every GitRepoConnector yield ``raws`` from discover() (no cloning)."""
    monkeypatch.setattr(GitRepoConnector, "discover", lambda self: iter(raws))


def _patch_discover_by_repo(
    monkeypatch: pytest.MonkeyPatch, mapping: dict[str, object]
) -> None:
    """Route discover() per repo via the connector's ``only`` filter.

    ``mapping`` maps 'owner/name' → either a list of RawChallenge to yield, or an
    Exception instance to raise (simulating a clone/walk failure for that repo).
    """

    def _discover(self):
        for repo in self.only or set():
            item = mapping.get(repo)
            if isinstance(item, Exception):
                raise item
            if item is not None:
                return iter(item)  # type: ignore[arg-type]
        return iter([])

    monkeypatch.setattr(GitRepoConnector, "discover", _discover)


def test_stage_repo_isolated_materializes_and_drops_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    work_root = data_dir / "work"
    _patch_discover(monkeypatch, [_make_raw(tmp_path, "acme/pwn", "target_practice")])

    result, work_dir = pipeline.stage_repo_isolated(
        "acme/pwn",
        data_dir=data_dir,
        work_root=work_root,
        max_repo_size_mb=4096,
        token="t",
    )

    # Staged (materialized, not yet published) with provenance + byte accounting.
    assert result.status == "staged"
    assert result.repo == "acme/pwn"
    assert result.commit_sha == _SHA
    assert result.n_challenges == 1
    # The challenge dir was packed into ONE per-challenge tarball, so this repo's own
    # corpus holds a single file (the archive), not the loose source tree.
    assert result.n_files == 1
    assert result.bytes > 0

    # The isolated work dir is returned; corpus/catalog live under it.
    slug = "acme__pwn"
    assert work_dir == work_root / slug
    corpus_root = work_dir / "corpus"
    catalog_root = work_dir / "catalog"

    # Raw clone dropped immediately; the per-repo work dir holds the materialized output.
    assert not (data_dir / "raw" / slug).exists()
    assert corpus_root.exists()
    shard = catalog_root / slug / "challenges.jsonl"
    assert shard.exists()

    # The batch corpus holds per-challenge .tar.gz archives, NOT loose source files.
    archives = list(corpus_root.rglob("*.tar.gz"))
    assert len(archives) == 1
    assert not list(corpus_root.rglob("chal.py"))  # no loose source tree remains
    assert not list(corpus_root.rglob("Dockerfile"))

    # The shard carries the HF-layout corpus_path (relative to staging_batch), now
    # pointing at the archive.
    records = [json.loads(line) for line in shard.read_text().splitlines() if line]
    assert records
    assert records[0]["corpus_path"].startswith("corpus/")
    assert records[0]["corpus_path"].endswith(".tar.gz")
    # The per-file browsable manifest is preserved in the catalog.
    assert {f["path"] for f in records[0]["files"]} >= {"chal.py", "Dockerfile"}


def test_stage_repo_archives_one_tarball_per_challenge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    work_root = data_dir / "work"
    # Two distinct challenges in one repo → two independent per-challenge tarballs.
    # Give each unique source bytes so they don't fingerprint-collapse in dedup.
    alpha = _make_raw(tmp_path, "acme/multi", "alpha")
    (Path(alpha.local_dir) / "unique.txt").write_text("alpha marker\n", encoding="utf-8")
    beta = _make_raw(tmp_path, "acme/multi", "beta")
    (Path(beta.local_dir) / "unique.txt").write_text("beta marker\n", encoding="utf-8")
    _patch_discover(monkeypatch, [alpha, beta])

    result, work_dir = pipeline.stage_repo_isolated(
        "acme/multi",
        data_dir=data_dir,
        work_root=work_root,
        max_repo_size_mb=4096,
        token="t",
    )
    corpus_root = work_dir / "corpus"
    catalog_root = work_dir / "catalog"

    assert result.status == "staged"
    assert result.n_challenges == 2
    # One .tar.gz per challenge; no loose source trees left in staging.
    archives = list(corpus_root.rglob("*.tar.gz"))
    assert len(archives) == 2
    assert not list(corpus_root.rglob("chal.py"))

    # Every catalog record's corpus_path points at a .tar.gz archive.
    shard = catalog_root / "acme__multi" / "challenges.jsonl"
    records = [json.loads(line) for line in shard.read_text().splitlines() if line]
    assert len(records) == 2
    assert all(r["corpus_path"].endswith(".tar.gz") for r in records)


def test_mirror_all_flushes_by_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    staging_batch = data_dir / "staging_batch"
    _patch_discover_by_repo(
        monkeypatch,
        {
            "acme/a": [_make_raw(tmp_path, "acme/a", "ca")],
            "acme/b": [_make_raw(tmp_path, "acme/b", "cb")],
            "acme/c": [_make_raw(tmp_path, "acme/c", "cc")],
        },
    )

    results = pipeline.mirror_all(
        ["acme/a", "acme/b", "acme/c"],
        data_dir=data_dir,
        dataset="acme/ds",
        publish=True,
        keep_local=False,
        max_repo_size_mb=4096,
        token="t",
        resume=False,
        batch_size=2,
    )

    # Two flushes for three repos (2 + 1), NOT one commit per repo.
    assert len(recorder.corpus_calls) == 2
    assert len(recorder.catalog_calls) == 2
    # First flush carried repos a+b, the second (final) carried c alone; staging is
    # cleaned between flushes so the second batch holds a single fresh shard.
    assert recorder.catalog_calls[0]["n_shards"] == 2
    assert recorder.catalog_calls[1]["n_shards"] == 1

    # Every repo recorded 'ok' — but only after its batch was published. Producers run
    # in parallel so completion order is not fixed; assert on sets, not sequence.
    assert all(r.status == "ok" for r in results)
    assert {r.repo for r in results} == {"acme/a", "acme/b", "acme/c"}

    # Staging cleaned after the final flush; the work root is gone too.
    assert not staging_batch.exists()
    assert not (data_dir / "work").exists()
    # Manifest durably records all three as ok (order depends on completion timing).
    manifest = data_dir / "mirror_state.jsonl"
    recs = [json.loads(x) for x in manifest.read_text().splitlines() if x]
    assert {(r["repo"], r["status"]) for r in recs} == {
        ("acme/a", "ok"),
        ("acme/b", "ok"),
        ("acme/c", "ok"),
    }


def test_mirror_all_records_ok_only_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fail the FIRST publish_corpus, succeed on the second.
    rec = _Recorder(raise_on_corpus_call=1)
    _install_recorder(monkeypatch, rec)
    data_dir = tmp_path / "data"
    staging_batch = data_dir / "staging_batch"
    _patch_discover_by_repo(
        monkeypatch,
        {
            "acme/a": [_make_raw(tmp_path, "acme/a", "ca")],
            "acme/b": [_make_raw(tmp_path, "acme/b", "cb")],
        },
    )

    # batch_size=1 → one flush per repo: first flush raises, second succeeds. workers=1
    # pins completion order to submission order so "first flush = acme/a" is deterministic
    # (the ok-only-after-publish invariant itself holds at any concurrency).
    results = pipeline.mirror_all(
        ["acme/a", "acme/b"],
        data_dir=data_dir,
        dataset="acme/ds",
        publish=True,
        keep_local=False,
        max_repo_size_mb=4096,
        token="t",
        resume=False,
        workers=1,
        batch_size=1,
    )

    # A's batch publish failed → 'failed' (NOT 'ok'); B's succeeded → 'ok'.
    by_repo = {r.repo: r.status for r in results}
    assert by_repo == {"acme/a": "failed", "acme/b": "ok"}

    # A failed publish still frees the staging tree (bytes re-clone next run).
    assert not staging_batch.exists()

    # Manifest never marked A 'ok'; a resume run would redo A.
    manifest = data_dir / "mirror_state.jsonl"
    recs = [json.loads(x) for x in manifest.read_text().splitlines() if x]
    assert {"repo": "acme/a", "status": "failed"} in [
        {"repo": r["repo"], "status": r["status"]} for r in recs
    ]
    assert not any(r["repo"] == "acme/a" and r["status"] == "ok" for r in recs)


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
    _patch_discover_by_repo(monkeypatch, {"acme/B": [_make_raw(tmp_path, "acme/B", "b")]})

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

    # A skipped (not re-emitted); B staged, flushed and marked ok.
    assert [r.repo for r in results] == ["acme/B"]
    assert results[0].status == "ok"
    assert len(recorder.corpus_calls) == 1  # exactly one batch commit for B

    # Manifest now records both A (pre-seeded) and B (appended after its flush).
    repos_in_manifest = [
        json.loads(line)["repo"] for line in manifest.read_text().splitlines() if line
    ]
    assert repos_in_manifest == ["acme/A", "acme/B"]


def test_mirror_all_failed_repo_does_not_abort_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    _patch_discover_by_repo(
        monkeypatch,
        {
            "acme/bad": RuntimeError("clone timed out"),
            "acme/good": [_make_raw(tmp_path, "acme/good", "gchal")],
        },
    )

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

    # The bad repo failed but the loop continued and still staged+flushed the good one.
    by_repo = {r.repo: r.status for r in results}
    assert by_repo == {"acme/bad": "failed", "acme/good": "ok"}
    bad = next(r for r in results if r.repo == "acme/bad")
    assert bad.error and "clone timed out" in bad.error
    # The failing repo left no raw clone behind (cleanup ran in finally).
    assert not (data_dir / "raw" / "acme__bad").exists()


def test_mirror_all_publish_false_marks_ok_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    staging_batch = data_dir / "staging_batch"
    _patch_discover_by_repo(monkeypatch, {"acme/a": [_make_raw(tmp_path, "acme/a", "ca")]})

    results = pipeline.mirror_all(
        ["acme/a"],
        data_dir=data_dir,
        dataset="acme/ds",
        publish=False,
        keep_local=False,
        max_repo_size_mb=4096,
        token=None,
        resume=False,
    )

    # Staged repo still marked ok, but no publish call was made.
    assert [r.status for r in results] == ["ok"]
    assert recorder.corpus_calls == []
    assert recorder.catalog_calls == []
    # Staging cleaned even offline.
    assert not staging_batch.exists()


def test_mirror_all_keep_local_retains_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    staging_batch = data_dir / "staging_batch"
    _patch_discover_by_repo(monkeypatch, {"acme/a": [_make_raw(tmp_path, "acme/a", "ca")]})

    results = pipeline.mirror_all(
        ["acme/a"],
        data_dir=data_dir,
        dataset="acme/ds",
        publish=False,
        keep_local=True,
        max_repo_size_mb=4096,
        token=None,
        resume=False,
    )

    assert [r.status for r in results] == ["ok"]
    # keep_local retains the batch staging tree after the flush.
    assert (staging_batch / "corpus").exists()
    # Raw clone is still always dropped.
    assert not (data_dir / "raw" / "acme__a").exists()


def test_stage_repo_disk_guard_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections import namedtuple

    data_dir = tmp_path / "data"
    work_root = data_dir / "work"
    _patch_discover(monkeypatch, [_make_raw(tmp_path, "acme/big", "chal")])

    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        pipeline.shutil, "disk_usage", lambda _p: usage(100, 99, 1 * 1024**3)
    )

    result, work_dir = pipeline.stage_repo_isolated(
        "acme/big",
        data_dir=data_dir,
        work_root=work_root,
        max_repo_size_mb=4096,
        token="t",
    )

    assert result.status == "skipped"
    assert work_dir is None
    assert result.error and "low disk" in result.error
    # Guard fired before any clone/materialize.
    assert not (data_dir / "raw" / "acme__big").exists()
    assert not (work_root / "acme__big").exists()


def test_mirror_all_parallel_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
) -> None:
    data_dir = tmp_path / "data"
    # 10 repos, each one challenge; staged concurrently by 4 workers, flushed in 2 batches
    # of 5. Repos share the same origin/event/year corpus prefix, so this also exercises
    # the consumer's tree-merge of independent per-repo work dirs into the one batch tree.
    repos = [f"acme/r{i}" for i in range(10)]
    mapping = {r: [_make_raw(tmp_path, r, f"c{i}")] for i, r in enumerate(repos)}
    _patch_discover_by_repo(monkeypatch, mapping)

    results = pipeline.mirror_all(
        repos,
        data_dir=data_dir,
        dataset="acme/ds",
        publish=True,
        keep_local=False,
        max_repo_size_mb=4096,
        token="t",
        resume=False,
        workers=4,
        batch_size=5,
    )

    # All 10 processed and marked 'ok' after their batch published — regardless of the
    # order the parallel producers happened to finish in.
    assert len(results) == 10
    assert {r.repo for r in results} == set(repos)
    assert all(r.status == "ok" for r in results)
    assert sum(r.n_challenges for r in results) == 10

    # Exactly two batch commits, five repos each (count-triggered flush + final flush).
    assert len(recorder.corpus_calls) == 2
    assert len(recorder.catalog_calls) == 2
    assert [c["n_shards"] for c in recorder.catalog_calls] == [5, 5]

    # Manifest durably records all ten as 'ok'.
    manifest = data_dir / "mirror_state.jsonl"
    recs = [json.loads(x) for x in manifest.read_text().splitlines() if x]
    assert {r["repo"] for r in recs} == set(repos)
    assert all(r["status"] == "ok" for r in recs)

    # No leftover isolated work dirs and no leftover staging after a successful run.
    assert not (data_dir / "work").exists()
    assert not (data_dir / "staging_batch").exists()


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
