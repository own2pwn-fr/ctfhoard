"""Publish the materialized corpus and catalog to a Hugging Face dataset.

GitHub carries the code + normalized catalog + text corpus, but the *complete*
hard-copy corpus (including binaries such as firmware images, pcaps and archives)
is too heavy for a plain git repo. This module mirrors ``data/corpus/`` to a public
Hugging Face dataset repo (default :data:`HF_CORPUS_DATASET`) so the bytes live where
big binary blobs belong, and pushes the JSONL catalog alongside them so the dataset is
self-describing.

Network access is confined to :func:`publish_corpus` / :func:`publish_catalog`; the
reporting helper :func:`corpus_stats` and every ``dry_run=True`` path are pure-local and
safe offline. The authentication token is read from the ``token`` argument or the
``HF_TOKEN`` environment variable.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path

from loguru import logger

HF_CORPUS_DATASET = "own2pwn-fr/ctfhoard-corpus"

#: How many times to retry a folder upload before giving up.
_MAX_UPLOAD_ATTEMPTS = 6
#: Hard ceiling per upload attempt. HF's LFS/Xet upload can STALL — many idle
#: connections, CPU and disk I/O at zero, no internal timeout — and hang the whole
#: marathon forever. We run each attempt in a child process and terminate it past this
#: deadline, then retry (upload is incremental, so the next attempt resumes).
_UPLOAD_TIMEOUT_S = 1800.0


def _upload_worker(queue, token: str, kwargs: dict) -> None:
    """Child-process entrypoint: perform one ``upload_folder`` and report via queue."""
    try:
        from huggingface_hub import HfApi

        url = HfApi(token=token).upload_folder(**kwargs)
        queue.put(("ok", str(url)))
    except Exception as exc:  # noqa: BLE001 — ship the failure back to the parent
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


def _upload_folder_with_retry(resolved_token: str, **kwargs) -> str:
    """``HfApi.upload_folder`` with a hard per-attempt timeout + retry.

    Two failure modes are handled: a transient error (connection reset → client
    closed) that raises, and a silent STALL where the upload hangs with no progress
    and no internal timeout. Each attempt runs in a spawned child process joined with
    :data:`_UPLOAD_TIMEOUT_S`; a hung child is terminated. ``upload_folder`` is
    incremental (diffs the remote, sends only missing files), so a retry resumes where
    the last attempt stopped and successive tries converge. Only after all attempts are
    exhausted do we raise.
    """
    ctx = multiprocessing.get_context("spawn")
    last_err = "unknown upload failure"
    for attempt in range(1, _MAX_UPLOAD_ATTEMPTS + 1):
        queue: multiprocessing.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_upload_worker, args=(queue, resolved_token, kwargs), daemon=True
        )
        proc.start()
        proc.join(_UPLOAD_TIMEOUT_S)
        if proc.is_alive():  # stalled — kill it and retry
            proc.terminate()
            proc.join(30)
            if proc.is_alive():
                proc.kill()
            last_err = f"upload hung > {_UPLOAD_TIMEOUT_S:.0f}s (terminated)"
        else:
            try:
                status, payload = queue.get_nowait()
            except Exception:  # noqa: BLE001
                status, payload = "err", f"worker exited ({proc.exitcode}) with no result"
            if status == "ok":
                return payload
            last_err = payload
        if attempt < _MAX_UPLOAD_ATTEMPTS:
            wait = min(60.0, 2.0**attempt)
            logger.warning(
                "HF upload attempt {}/{} failed ({}); retrying in {}s",
                attempt,
                _MAX_UPLOAD_ATTEMPTS,
                last_err,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"HF upload failed after {_MAX_UPLOAD_ATTEMPTS} attempts: {last_err}"
    )


class HFTokenError(RuntimeError):
    """Raised when no Hugging Face token is available for an upload."""


def _resolve_token(token: str | None) -> str:
    """Return an explicit token, else ``HF_TOKEN`` from the environment, else raise."""
    resolved = token or os.environ.get("HF_TOKEN")
    if not resolved:
        raise HFTokenError(
            "no Hugging Face token: pass token= or set the HF_TOKEN environment variable"
        )
    return resolved


def corpus_stats(corpus_dir: Path) -> dict:
    """Summarize a corpus tree for reporting, without touching the network.

    Returns ``{"files": int, "total_bytes": int, "by_extension_top": [(ext, count), ...]}``
    where ``by_extension_top`` lists the most common file extensions (``""`` for files
    with no suffix). An empty or missing directory yields zeroed counts rather than an error.
    """
    corpus_dir = Path(corpus_dir)
    files = 0
    total_bytes = 0
    ext_counts: Counter[str] = Counter()
    if corpus_dir.exists():
        for path in corpus_dir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            files += 1
            total_bytes += path.stat().st_size
            ext_counts[path.suffix.lower()] += 1
    return {
        "files": files,
        "total_bytes": total_bytes,
        "by_extension_top": ext_counts.most_common(10),
    }


def _format_stats(stats: dict) -> str:
    """One-line human summary of :func:`corpus_stats` output."""
    exts = ", ".join(f"{ext or '<none>'}={n}" for ext, n in stats["by_extension_top"])
    return (
        f"{stats['files']} files, {stats['total_bytes']} bytes"
        + (f" [{exts}]" if exts else "")
    )


def publish_corpus(
    corpus_dir: Path,
    *,
    dataset: str = HF_CORPUS_DATASET,
    path_in_repo: str = "corpus",
    commit_message: str | None = None,
    token: str | None = None,
    allow_patterns=None,
    dry_run: bool = False,
) -> str:
    """Upload a local corpus tree to a Hugging Face dataset repo.

    Uploads ``corpus_dir`` under ``path_in_repo`` in ``dataset`` using
    :meth:`huggingface_hub.HfApi.upload_folder`, which chunks large/binary files into an
    LFS-backed commit — well suited to the mixed text+binary corpus. ``allow_patterns``,
    if given, restricts which files are uploaded.

    With ``dry_run=True`` the function only walks ``corpus_dir`` and returns a textual
    report of what *would* be uploaded (file count + total bytes), never touching the
    network or requiring a token. Otherwise it returns the commit URL.
    """
    corpus_dir = Path(corpus_dir)
    stats = corpus_stats(corpus_dir)

    if stats["files"] == 0:
        msg = f"corpus dir {corpus_dir} is empty — nothing to upload"
        logger.warning(msg)
        return f"dry-run: {msg}" if dry_run else msg

    if dry_run:
        report = f"dry-run: would upload {_format_stats(stats)} to {dataset}/{path_in_repo}"
        logger.info(report)
        return report

    resolved_token = _resolve_token(token)
    commit_url = _upload_folder_with_retry(
        resolved_token,
        repo_id=dataset,
        repo_type="dataset",
        folder_path=str(corpus_dir),
        path_in_repo=path_in_repo,
        commit_message=commit_message or "Publish ctfhoard corpus",
        allow_patterns=allow_patterns,
    )
    logger.info("uploaded corpus ({}) to {}", _format_stats(stats), commit_url)
    return str(commit_url)


def publish_catalog(
    catalog_dir: Path,
    *,
    dataset: str = HF_CORPUS_DATASET,
    token: str | None = None,
    dry_run: bool = False,
) -> str:
    """Push the JSONL catalog shards to the dataset under ``catalog/``.

    Mirrors ``catalog_dir`` (only ``*.jsonl`` shards) into the dataset so the published
    corpus is self-describing. ``dry_run=True`` reports what would be pushed without any
    network access; otherwise returns the commit URL.
    """
    catalog_dir = Path(catalog_dir)
    shards = sorted(catalog_dir.rglob("*.jsonl")) if catalog_dir.exists() else []

    if not shards:
        msg = f"no JSONL catalog shards under {catalog_dir} — nothing to upload"
        logger.warning(msg)
        return f"dry-run: {msg}" if dry_run else msg

    total_bytes = sum(p.stat().st_size for p in shards)
    if dry_run:
        report = (
            f"dry-run: would upload {len(shards)} catalog shard(s), "
            f"{total_bytes} bytes to {dataset}/catalog"
        )
        logger.info(report)
        return report

    resolved_token = _resolve_token(token)
    commit_url = _upload_folder_with_retry(
        resolved_token,
        repo_id=dataset,
        repo_type="dataset",
        folder_path=str(catalog_dir),
        path_in_repo="catalog",
        commit_message="Publish ctfhoard catalog",
        allow_patterns=["*.jsonl", "**/*.jsonl"],
    )
    logger.info(
        "uploaded {} catalog shard(s) ({} bytes) to {}", len(shards), total_bytes, commit_url
    )
    return str(commit_url)
