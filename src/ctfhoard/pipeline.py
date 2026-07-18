"""Disk-bounded streaming mirror of many repos to the Hugging Face dataset.

The plain ``ingest`` command clones every seed repo, materializes the *whole* corpus
on disk, then publishes it in one shot — which needs enough local disk to hold all
repos at once. This module streams instead: it processes ONE repo end-to-end
(clone → normalize → dedup → materialize → publish that repo's corpus subtree +
catalog shard to HF), then deletes that repo's raw clone and corpus staging before
moving to the next. Peak local disk stays bounded to roughly a single repo, so the
mirror scales to the whole long tail of CTF repositories.

Two invariants make this safe and resumable:

* **Isolation** — each repo materializes into its own ``staging/<slug>/corpus`` tree
  whose layout (``corpus/<origin>/<event>/<year>/<slug>__id``) matches the HF dataset
  exactly, so :meth:`huggingface_hub.HfApi.upload_folder` only diffs/sends new files
  while the local bytes remain in a single deletable directory.
* **Durability** — a resume manifest (``mirror_state.jsonl``) records each completed
  repo the moment it finishes, so a crash mid-run resumes from where it stopped and
  never re-publishes a repo already marked ``ok``.

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

Status = Literal["ok", "failed", "skipped", "too_big"]


class RepoResult(BaseModel):
    """Outcome of mirroring a single repo (one manifest/summary row).

    ``status`` is ``ok`` when the repo was cloned, materialized and (optionally)
    published; ``skipped`` when the disk-guard floor was hit before cloning;
    ``too_big`` when the repo's known size exceeds the size cap; ``failed`` when any
    step raised (the error is captured, the loop continues). ``n_files``/``bytes``
    describe the materialized corpus subtree that was (or would be) published.
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


def mirror_repo(
    repo: str,
    *,
    data_dir: Path,
    dataset: str,
    publish: bool,
    keep_local: bool,
    max_repo_size_mb: int,
    token: str | None,
) -> RepoResult:
    """Mirror one repo end-to-end, keeping local disk bounded to this repo alone.

    Clones ``repo`` under ``data_dir/raw``, walks/normalizes/dedups it, materializes
    hard copies into an isolated per-repo ``staging/<slug>/corpus`` tree (whose layout
    matches the HF dataset), writes the small catalog shard under ``catalog/<slug>``,
    optionally publishes the corpus subtree + shard to ``dataset``, then ALWAYS deletes
    the raw clone (and, unless ``keep_local``, the staging tree) — even on failure — so
    the next repo starts from a clean, bounded disk. Never raises: any error becomes a
    ``failed`` :class:`RepoResult` so the caller's loop continues.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    slug = _repo_dirname(repo)
    raw_workdir = data_dir / "raw"
    clone_dir = raw_workdir / slug
    staging_dir = data_dir / "staging" / slug
    corpus_root = staging_dir / "corpus"
    catalog_dir = data_dir / "catalog" / slug

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
        disc = data_dir / "discovered_repos.jsonl"
        connector = GitRepoConnector(
            workdir=raw_workdir,
            only=[repo],
            max_repo_size_mb=max_repo_size_mb,
            discovered_path=disc if disc.exists() else None,
        )

        # Clone → walk → normalize → dedup.
        challenges = [normalize(raw) for raw in connector.discover()]
        challenges = _dedup(challenges)
        commit_sha = _commit_sha_of(challenges)

        # Materialize each challenge into the isolated per-repo staging corpus. Using
        # repo_root=staging makes corpus_path come out as 'corpus/<origin>/...', the
        # exact HF layout, while the bytes stay under a single deletable directory.
        repo_root = staging_dir.resolve()
        for ch in challenges:
            raw_dir = (
                Path(ch.corpus_path)
                if ch.corpus_path and Path(ch.corpus_path).exists()
                else None
            )
            # client=None: repo writeups are in-repo files copied with the sources; no
            # external link is followed here.
            materialize_challenge(
                ch, corpus_root, raw_dir=raw_dir, client=None, repo_root=repo_root
            )

        # Write the (small, KEPT) catalog shard for this repo.
        writer = CatalogWriter(catalog_dir)
        writer.extend(challenges)
        writer.write_jsonl()

        stats = hf.corpus_stats(corpus_root)

        if publish:
            hf.publish_corpus(
                corpus_root, dataset=dataset, path_in_repo="corpus", token=token
            )
            hf.publish_catalog(catalog_dir, dataset=dataset, token=token)

        return RepoResult(
            repo=repo,
            commit_sha=commit_sha,
            n_challenges=len(challenges),
            n_files=stats["files"],
            bytes=stats["total_bytes"],
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001 — one bad repo must never abort the run
        logger.warning("mirror_repo {} failed: {}", repo, exc)
        return RepoResult(
            repo=repo,
            commit_sha=commit_sha,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        # ALWAYS free the raw clone; drop the staging corpus unless the caller keeps it.
        # A mid-repo failure still reclaims disk here before the next repo runs.
        shutil.rmtree(clone_dir, ignore_errors=True)
        if not keep_local:
            shutil.rmtree(staging_dir, ignore_errors=True)


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
) -> list[RepoResult]:
    """Mirror many repos one at a time, streaming and resumable.

    Loads the resume manifest at ``data_dir/mirror_state.jsonl``; when ``resume`` is set,
    repos already recorded ``ok`` are skipped. Each repo is mirrored by :func:`mirror_repo`
    (which bounds local disk to one repo) and its result appended to the manifest the
    moment it completes, so a crash resumes cleanly. Returns every result (including
    skips of already-done repos are NOT re-emitted — only repos actually processed here).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = data_dir / "mirror_state.jsonl"

    done_ok = _load_done_ok(manifest) if resume else set()
    results: list[RepoResult] = []
    total = len(repos)

    for i, repo in enumerate(repos, start=1):
        if resume and repo in done_ok:
            logger.info("[{}/{}] {} → skip (already ok)", i, total, repo)
            continue

        result = mirror_repo(
            repo,
            data_dir=data_dir,
            dataset=dataset,
            publish=publish,
            keep_local=keep_local,
            max_repo_size_mb=max_repo_size_mb,
            token=token,
        )
        results.append(result)
        _append_manifest(manifest, result, ts=datetime.now(UTC))

        free = shutil.disk_usage(data_dir).free
        logger.info(
            "[{}/{}] {} → {} ({} challenges, {:.1f} MB, free disk {:.1f} GB)",
            i,
            total,
            repo,
            result.status,
            result.n_challenges,
            result.bytes / 1024**2,
            free / 1024**3,
        )

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
