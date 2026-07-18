"""Disk-bounded, commit-batched mirror of many repos to the Hugging Face dataset.

The plain ``ingest`` command clones every seed repo, materializes the *whole* corpus
on disk, then publishes it in one shot — which needs enough local disk to hold all
repos at once. Publishing one repo at a time bounds disk but pays TWO HF commits per
repo; at the scale of thousands of repos Hugging Face rate-limits commits (HTTP 429 →
multi-minute back-offs) and the marathon stretches into weeks.

This module keeps disk bounded *and* cuts commit count ~50× by batching. It stages
MANY repos into ONE shared staging tree (``data_dir/staging_batch``): each repo is
cloned, walked, normalized, dedup'd and materialized into the shared
``staging_batch/corpus`` tree, its raw clone is deleted immediately (the materialized
copy is smaller), and its catalog shard is written under ``staging_batch/catalog``.
Once the batch reaches ``batch_size`` repos or ``batch_max_mb`` on disk, the *whole*
batch is published in a single corpus commit + a single catalog commit, then the
staging tree is deleted before the next batch starts. Peak local disk stays bounded to
one batch, and HF sees two commits per batch instead of two per repo.

Two invariants make this safe and resumable:

* **Isolation** — the shared staging tree's layout
  (``corpus/<origin>/<event>/<year>/<slug>__id``) matches the HF dataset exactly, so
  :meth:`huggingface_hub.HfApi.upload_folder` only diffs/sends new files while the
  local bytes remain in a single deletable directory.
* **Durability** — a resume manifest (``mirror_state.jsonl``) records each repo as
  ``ok`` ONLY after the batch it belongs to has been published to HF. A crash mid-batch
  (before the flush) leaves those repos un-``ok``, so the next run simply re-does them —
  never a silent gap, never a double publish of an already-``ok`` repo.

Every per-repo body is wrapped so a clone timeout, walk error, or publish failure is
recorded as a failed :class:`RepoResult` and NEVER aborts the surrounding loop.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from ctfhoard import hf
from ctfhoard.connectors.git_repo import GitRepoConnector
from ctfhoard.corpus import materialize_challenge
from ctfhoard.dedup import cluster_by_fingerprint, merge_cluster
from ctfhoard.discover import load_discovered
from ctfhoard.mirror import _repo_dirname
from ctfhoard.normalize import normalize
from ctfhoard.schema import Challenge
from ctfhoard.storage import CatalogWriter

#: Safety floor: refuse to start a new clone when the data dir has less free space
#: than this (a single large repo + its materialized corpus must comfortably fit).
_MIN_FREE_BYTES: int = 5 * 1024**3  # 5 GiB

#: Seed-file sections flattened, in authority order, into one repo list.
_SEED_SECTIONS = ("official_sources", "community_archives", "writeup_archives")

#: Repo-root ``seeds/official_repos.yaml`` (pipeline.py -> ctfhoard -> src -> root).
_DEFAULT_SEEDS = Path(__file__).resolve().parents[2] / "seeds" / "official_repos.yaml"

Status = Literal["ok", "failed", "skipped", "too_big", "staged"]


class RepoResult(BaseModel):
    """Outcome of mirroring a single repo (one manifest/summary row).

    ``status`` is ``staged`` when the repo was cloned and materialized into the current
    batch but not yet published; it becomes ``ok`` once that batch is committed to HF,
    or ``failed`` if the batch publish fails (the repo re-clones next run). ``skipped``
    means the disk-guard floor was hit before cloning; ``too_big`` means the repo's
    known size exceeds the size cap; ``failed`` also covers any step that raised while
    staging (the error is captured, the loop continues). ``n_files``/``bytes`` describe
    the materialized corpus subtree this repo contributed to the batch.
    """

    repo: str = Field(description="'owner/name' identifier on GitHub.")
    commit_sha: str | None = Field(
        default=None, description="Pinned HEAD the challenges were mirrored at."
    )
    n_challenges: int = Field(default=0, ge=0, description="Deduplicated challenges kept.")
    n_files: int = Field(default=0, ge=0, description="Files in the materialized corpus.")
    bytes: int = Field(default=0, ge=0, description="Total bytes of the materialized corpus.")
    status: Status = Field(description="ok | failed | skipped | too_big.")
    error: str | None = Field(default=None, description="Failure/skip reason, if any.")


def _dedup(challenges: list[Challenge]) -> list[Challenge]:
    """Collapse fingerprint-identical mirrors into canonical records.

    Mirrors :func:`ctfhoard.cli._dedup`: within a repo the same challenge can appear
    twice (e.g. a challenge and its re-hosted copy); we cluster by content fingerprint,
    keep the most authoritative member, and mark the losers ``duplicate_of`` it.
    """
    clusters = cluster_by_fingerprint(challenges)
    canonical: list[Challenge] = []
    for members in clusters.values():
        merged = merge_cluster(members)
        for m in members:
            if m.id != merged.id:
                m.duplicate_of = merged.id
        canonical.append(merged)
    return canonical


def _commit_sha_of(challenges: list[Challenge]) -> str | None:
    """First pinned commit SHA found across the challenges' provenance sources."""
    for ch in challenges:
        for src in ch.sources:
            if src.commit_sha:
                return src.commit_sha
    return None


def _discovered_size_kb(data_dir: Path, repo: str) -> int | None:
    """Repo size (KB) from a local ``discovered_repos.jsonl``, or None if unknown.

    Used only for the ``too_big`` pre-check: curated seed repos are never size-capped,
    so a repo whose size we cannot cheaply know is simply cloned.
    """
    disc = data_dir / "discovered_repos.jsonl"
    if not disc.exists():
        return None
    try:
        for cand in load_discovered(disc):
            if cand.full_name == repo:
                return cand.size_kb
    except OSError:
        return None
    return None


def stage_repo(
    repo: str,
    *,
    data_dir: Path,
    batch_corpus_root: Path,
    batch_catalog_root: Path,
    max_repo_size_mb: int,
    token: str | None,
    discovered_path: Path | None = None,
) -> RepoResult:
    """Clone one repo and materialize it INTO the current shared batch (no publish).

    Clones ``repo`` under ``data_dir/raw``, walks/normalizes/dedups it, materializes
    hard copies into the SHARED ``batch_corpus_root`` (whose layout matches the HF
    dataset), writes the repo's catalog shard under ``batch_catalog_root/<slug>``, then
    ALWAYS deletes the raw clone — even on failure — so disk stays bounded to the
    materialized batch alone. It does NOT publish and does NOT touch the shared batch
    staging (the caller flushes the whole batch later).

    Never raises: a clone timeout, walk error or any other exception becomes a
    ``failed`` :class:`RepoResult` so the caller's loop continues. A successful stage
    returns status ``staged`` (materialized-into-batch, not yet published); the
    disk-guard floor yields ``skipped`` and the size cap yields ``too_big``, both
    before any clone.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    batch_corpus_root = Path(batch_corpus_root)
    batch_catalog_root = Path(batch_catalog_root)

    slug = _repo_dirname(repo)
    raw_workdir = data_dir / "raw"
    clone_dir = raw_workdir / slug

    # Disk guard: refuse to start a clone that could exhaust the disk mid-run.
    free = shutil.disk_usage(data_dir).free
    if free < _MIN_FREE_BYTES:
        return RepoResult(
            repo=repo,
            status="skipped",
            error=(
                f"low disk: {free / 1024**3:.1f} GiB free below "
                f"{_MIN_FREE_BYTES / 1024**3:.0f} GiB floor"
            ),
        )

    # Cheap size pre-check for open-endedly discovered repos (seeds are never capped).
    size_kb = _discovered_size_kb(data_dir, repo)
    if size_kb is not None and size_kb > max_repo_size_mb * 1024:
        return RepoResult(
            repo=repo,
            status="too_big",
            error=f"{size_kb} KB exceeds {max_repo_size_mb} MB cap",
        )

    commit_sha: str | None = None
    try:
        # A discovered repo is not in the curated seeds, so let the connector treat the
        # discovered-repo list as extra seeds when it exists (harmless for seed repos:
        # the ``only`` filter + known-repo dedup keep the walk to exactly this repo).
        disc = (
            discovered_path
            if discovered_path is not None and Path(discovered_path).exists()
            else None
        )
        connector = GitRepoConnector(
            workdir=raw_workdir,
            only=[repo],
            max_repo_size_mb=max_repo_size_mb,
            discovered_path=disc,
        )

        # Clone → walk → normalize → dedup.
        challenges = [normalize(raw) for raw in connector.discover()]
        challenges = _dedup(challenges)
        commit_sha = _commit_sha_of(challenges)

        # Measure the batch corpus before/after so this repo's own file/byte
        # contribution can be reported (the batch tree is shared and accumulates).
        before = hf.corpus_stats(batch_corpus_root)

        # Materialize each challenge into the SHARED batch corpus. Using
        # repo_root=batch_corpus_root.parent (= staging_batch) makes corpus_path come
        # out as 'corpus/<origin>/...', the exact HF layout, while all batch bytes stay
        # under one deletable directory.
        repo_root = batch_corpus_root.parent.resolve()
        for ch in challenges:
            raw_dir = (
                Path(ch.corpus_path)
                if ch.corpus_path and Path(ch.corpus_path).exists()
                else None
            )
            # client=None: repo writeups are in-repo files copied with the sources; no
            # external link is followed here.
            materialize_challenge(
                ch, batch_corpus_root, raw_dir=raw_dir, client=None, repo_root=repo_root
            )

        after = hf.corpus_stats(batch_corpus_root)

        # Write this repo's catalog shard into the shared batch catalog tree.
        writer = CatalogWriter(batch_catalog_root / slug)
        writer.extend(challenges)
        writer.write_jsonl()

        return RepoResult(
            repo=repo,
            commit_sha=commit_sha,
            n_challenges=len(challenges),
            n_files=after["files"] - before["files"],
            bytes=after["total_bytes"] - before["total_bytes"],
            status="staged",
        )
    except Exception as exc:  # noqa: BLE001 — one bad repo must never abort the run
        logger.warning("stage_repo {} failed: {}", repo, exc)
        return RepoResult(
            repo=repo,
            commit_sha=commit_sha,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        # ALWAYS free the raw clone immediately: the materialized copy in the batch is
        # smaller, so dropping the clone keeps peak disk bounded to one batch. The
        # shared batch staging is NEVER touched here — the caller flushes it.
        shutil.rmtree(clone_dir, ignore_errors=True)


def _load_done_ok(manifest: Path) -> set[str]:
    """Return the set of repos already recorded ``ok`` in the resume manifest."""
    done: set[str] = set()
    if not manifest.exists():
        return done
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("status") == "ok" and rec.get("repo"):
                done.add(str(rec["repo"]))
    return done


def _append_manifest(manifest: Path, result: RepoResult, *, ts: datetime) -> None:
    """Append one completed repo to the resume manifest, flushed immediately.

    Written as a compact record (repo, status, commit_sha, n_challenges, bytes, ts) and
    flushed immediately so a crash mid-run leaves a durable, resumable trail.
    """
    manifest.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "repo": result.repo,
        "status": result.status,
        "commit_sha": result.commit_sha,
        "n_challenges": result.n_challenges,
        "bytes": result.bytes,
        "ts": ts.isoformat(),
    }
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record))
        fh.write("\n")
        fh.flush()


def _flush_batch(
    batch_staged: list[RepoResult],
    *,
    data_dir: Path,
    dataset: str,
    publish: bool,
    keep_local: bool,
    token: str | None,
    batch_corpus_root: Path,
    batch_catalog_root: Path,
    manifest: Path,
) -> list[RepoResult]:
    """Publish one accumulated batch in a single corpus + catalog commit, then finalize.

    ``batch_staged`` holds the repos materialized into the shared staging tree since the
    last flush. When ``publish`` is set and the batch corpus is non-empty, the WHOLE
    batch is uploaded as ONE corpus commit + ONE catalog commit (this is the ~50×
    commit-count reduction versus per-repo publishing).

    * On success (or ``publish=False``): every staged repo is recorded ``ok`` in the
      manifest — this is the ONLY place a repo becomes ``ok``, so a repo is durably done
      strictly *after* its bytes reached HF.
    * On publish failure: every staged repo is recorded ``failed`` so the next run
      re-clones and re-stages it.

    The shared staging tree is deleted afterwards — always on failure (bytes will be
    re-cloned) and on success unless ``keep_local`` — so peak disk stays bounded to one
    batch. Returns the finalized results (with updated status). A no-op on an empty batch.
    """
    if not batch_staged:
        return []

    staging_batch = Path(data_dir) / "staging_batch"
    stats = hf.corpus_stats(batch_corpus_root)

    published_ok = True
    commit_url: str | None = None
    if publish and stats["files"] > 0:
        try:
            commit_url = hf.publish_corpus(
                batch_corpus_root, dataset=dataset, path_in_repo="corpus", token=token
            )
            hf.publish_catalog(batch_catalog_root, dataset=dataset, token=token)
        except Exception as exc:  # noqa: BLE001 — a failed batch retries next run
            logger.warning(
                "flush publish failed ({}): {} repo(s) will re-stage next run",
                exc,
                len(batch_staged),
            )
            published_ok = False

    # Finalize: 'ok' only once the batch is on HF; 'failed' means redo next run.
    ts = datetime.now(UTC)
    finalized: list[RepoResult] = []
    for r in batch_staged:
        r.status = "ok" if published_ok else "failed"
        if not published_ok:
            r.error = "batch publish failed; re-clones next run"
        _append_manifest(manifest, r, ts=ts)
        finalized.append(r)

    # Reclaim the shared staging tree: always when the publish failed (they re-clone),
    # and on success unless the caller keeps local bytes for inspection.
    if not published_ok or not keep_local:
        shutil.rmtree(staging_batch, ignore_errors=True)

    logger.info(
        "flush: published {} repos, {} files/{:.1f} MB, {}",
        len(finalized),
        stats["files"],
        stats["total_bytes"] / 1024**2,
        commit_url or ("skipped (empty batch)" if publish else "publish disabled"),
    )
    return finalized


def mirror_all(
    repos: list[str],
    *,
    data_dir: Path,
    dataset: str,
    publish: bool,
    keep_local: bool,
    max_repo_size_mb: int,
    token: str | None,
    resume: bool,
    batch_size: int = 50,
    batch_max_mb: int = 15000,
) -> list[RepoResult]:
    """Mirror many repos in commit-batches, disk-bounded and resumable.

    Loads the resume manifest at ``data_dir/mirror_state.jsonl``; when ``resume`` is set,
    repos already recorded ``ok`` are skipped. Each remaining repo is staged into a
    shared batch by :func:`stage_repo`; once the batch reaches ``batch_size`` staged
    repos OR ``batch_max_mb`` MB on disk, :func:`_flush_batch` publishes the whole batch
    in one corpus + one catalog commit and marks its repos ``ok``. Any remaining staged
    repos are flushed at the end.

    Resume correctness: a repo is recorded ``ok`` ONLY after its batch is published to
    HF. If the process dies mid-batch (before a flush), those repos are never ``ok`` and
    get redone next run — no gaps, no double publishing. Non-staged outcomes
    (``failed``/``skipped``/``too_big``) are recorded as-is immediately. Returns every
    result actually processed here (already-``ok`` skips are not re-emitted).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "mirror_state.jsonl"

    staging_batch = data_dir / "staging_batch"
    batch_corpus_root = staging_batch / "corpus"
    batch_catalog_root = staging_batch / "catalog"
    discovered_path = data_dir / "discovered_repos.jsonl"

    # A leftover staging tree from a crashed run is UNPUBLISHED (its repos are not 'ok'):
    # drop it so the first batch starts clean; those repos are re-staged from scratch.
    shutil.rmtree(staging_batch, ignore_errors=True)

    done_ok = _load_done_ok(manifest) if resume else set()
    results: list[RepoResult] = []
    batch_staged: list[RepoResult] = []
    batch_bytes = 0
    total = len(repos)

    def _flush() -> None:
        nonlocal batch_staged, batch_bytes
        finalized = _flush_batch(
            batch_staged,
            data_dir=data_dir,
            dataset=dataset,
            publish=publish,
            keep_local=keep_local,
            token=token,
            batch_corpus_root=batch_corpus_root,
            batch_catalog_root=batch_catalog_root,
            manifest=manifest,
        )
        results.extend(finalized)
        batch_staged = []
        batch_bytes = 0

    for i, repo in enumerate(repos, start=1):
        if resume and repo in done_ok:
            logger.info("[{}/{}] {} → skip (already ok)", i, total, repo)
            continue

        result = stage_repo(
            repo,
            data_dir=data_dir,
            batch_corpus_root=batch_corpus_root,
            batch_catalog_root=batch_catalog_root,
            max_repo_size_mb=max_repo_size_mb,
            token=token,
            discovered_path=discovered_path,
        )

        free = shutil.disk_usage(data_dir).free
        logger.info(
            "[{}/{}] {} → {} ({} ch, {:.1f} MB, free disk {:.1f} GB)",
            i,
            total,
            repo,
            result.status,
            result.n_challenges,
            result.bytes / 1024**2,
            free / 1024**3,
        )

        if result.status == "staged":
            batch_staged.append(result)
            batch_bytes += result.bytes
            if len(batch_staged) >= batch_size or batch_bytes >= batch_max_mb * 1024**2:
                _flush()
        else:
            # failed / skipped / too_big are terminal for this run: record as-is now.
            _append_manifest(manifest, result, ts=datetime.now(UTC))
            results.append(result)

    # Final flush of any repos staged since the last flush.
    _flush()
    return results


def resolve_repo_list(
    *,
    data_dir: Path,
    source: str,
    only: list[str] | None,
    seeds_path: Path | None = None,
    discovered_path: Path | None = None,
) -> list[str]:
    """Resolve the ordered, de-duplicated list of repos to mirror.

    ``only`` (if given) overrides everything to exactly those repos. Otherwise
    ``source`` selects: ``seeds`` → every repo in ``seeds/official_repos.yaml``;
    ``discovered`` → every ``full_name`` in ``discovered_repos.jsonl``; ``both`` →
    their union with seed repos first. Order is preserved and duplicates dropped.
    """
    if only:
        return _dedup_preserve(only)

    seeds_path = seeds_path or _DEFAULT_SEEDS
    discovered_path = discovered_path or (Path(data_dir) / "discovered_repos.jsonl")

    if source == "seeds":
        return _dedup_preserve(_seed_repos(seeds_path))
    if source == "discovered":
        return _dedup_preserve(_discovered_repos(discovered_path))
    if source == "both":
        return _dedup_preserve(
            [*_seed_repos(seeds_path), *_discovered_repos(discovered_path)]
        )
    raise ValueError(f"unknown source {source!r} (expected seeds|discovered|both)")


def _seed_repos(seeds_path: Path) -> list[str]:
    """Flatten every repo in the seed YAML, in authority-section order."""
    try:
        doc = yaml.safe_load(Path(seeds_path).read_text(encoding="utf-8")) or {}
    except OSError:
        return []
    repos: list[str] = []
    for section in _SEED_SECTIONS:
        for entry in doc.get(section, []) or []:
            if isinstance(entry, dict) and entry.get("repo"):
                repos.append(str(entry["repo"]))
    return repos


def _discovered_repos(discovered_path: Path) -> list[str]:
    """Full names of every candidate in a ``discovered_repos.jsonl`` (empty if absent)."""
    path = Path(discovered_path)
    if not path.exists():
        return []
    try:
        return [cand.full_name for cand in load_discovered(path)]
    except OSError:
        return []


def _dedup_preserve(items: list[str]) -> list[str]:
    """De-duplicate a list while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
