"""Materialize hard copies of challenges into the committed corpus tree.

The corpus is meant to *contain* the bytes, not point at them: every challenge's
source files and every writeup's content live on disk under ``data/corpus/`` and are
committed (binaries via Git LFS, text in plain git). This module is the single,
connector-independent place that guarantees that. Connectors just discover source
files (``local_dir``) and writeup URLs/text; :func:`materialize_challenge` copies the
sources in and fetches+writes the writeup bodies — following external links so a
CTFtime writeup that lives on someone's blog still ends up as a file in the repo.

Kept deliberately fault-tolerant: a single failed writeup fetch must never drop the
challenge or its sources.
"""

from __future__ import annotations

import gzip
import os
import re
import shutil
import tarfile
from pathlib import Path

from loguru import logger

from ctfhoard.dedup import is_source_file, sha256_bytes
from ctfhoard.http import PoliteClient
from ctfhoard.netguard import UnsafeUrlError, safe_get
from ctfhoard.normalize import slugify
from ctfhoard.schema import Challenge, FileEntry

# Map a response content-type / URL suffix to a file extension for the hard copy.
_CT_EXT = {
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "application/pdf": ".pdf",
    "application/json": ".json",
}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def challenge_relpath(challenge: Challenge) -> Path:
    """Deterministic corpus location for a challenge (stable across re-runs).

    ``<origin>/<event>/<year>/<challenge-slug>__<id8>`` — the id suffix disambiguates
    challenges that would otherwise slug-collide.
    """
    origin = challenge.sources[0].origin.value if challenge.sources else "unknown"
    event = slugify(challenge.event_name or "unknown-event") or "unknown-event"
    year = str(challenge.year or "0000")
    name = slugify(challenge.title) or "challenge"
    return Path(origin) / event / year / f"{name}__{challenge.id[:8]}"


def _writeup_ext(url: str | None, content_type: str | None) -> str:
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base in _CT_EXT:
            return _CT_EXT[base]
    if url:
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix in {".html", ".htm", ".md", ".txt", ".pdf", ".json"}:
            return ".html" if suffix == ".htm" else suffix
    return ".html"


def _copy_sources(raw_dir: Path, dest: Path) -> None:
    """Copy a connector's downloaded challenge tree into the corpus, minus VCS noise.

    Symlinks are never followed. A symlinked *file* is skipped outright; and any
    entry whose resolved target escapes ``raw_dir`` — e.g. reached by descending
    through a symlinked *directory* — is refused. This stops a malicious source
    tree from exfiltrating host files (``/etc/passwd``, cloud creds) into the
    published corpus.
    """
    root = raw_dir.resolve()
    # os.walk with followlinks=False so we NEVER descend into symlinked directories
    # (real repos, e.g. google-ctf, contain symlinked dirs like `exploit` -> ../..;
    # rglob would follow them, duplicating files and creating file/dir name clashes
    # that crash mkdir). Symlinked dirs are listed but not traversed; file symlinks
    # are skipped explicitly.
    for dirpath, dirnames, filenames in os.walk(raw_dir, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != ".git"]  # prune VCS dir from walk
        for fn in filenames:
            src = Path(dirpath) / fn
            if src.is_symlink():
                continue
            # Refuse anything that resolves outside raw_dir (defensive escape guard).
            try:
                if not src.resolve().is_relative_to(root):
                    continue
            except OSError:
                continue
            rel = src.relative_to(raw_dir)
            if ".git" in rel.parts:
                continue
            target = dest / rel
            # A name that is a file here but a directory elsewhere in the tree would
            # make mkdir/copy raise; skip such a conflicting entry rather than abort.
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
            except (FileExistsError, NotADirectoryError, OSError) as exc:
                logger.debug("skipped copying {}: {}", rel, exc)
                continue


def _stream_client(client: PoliteClient):
    """Return the httpx-style streaming client behind a ``PoliteClient`` wrapper.

    :func:`ctfhoard.netguard.safe_get` needs the streaming API (``.stream``);
    ``PoliteClient`` only exposes ``.get`` over its inner ``httpx.Client``. A client
    that already exposes ``.stream`` (a raw httpx client or a test fake) is used as
    is. Returns None when no streaming client can be found.
    """
    if hasattr(client, "stream"):
        return client
    return getattr(client, "_client", None)


def materialize_challenge(
    challenge: Challenge,
    corpus_root: Path,
    *,
    raw_dir: Path | None = None,
    client: PoliteClient | None = None,
    repo_root: Path | None = None,
) -> Challenge:
    """Write hard copies of a challenge's sources and writeups under ``corpus_root``.

    * ``raw_dir`` — the connector's local download dir (its files are copied in). If
      None, the challenge is source-less (e.g. a pure-writeup/metadata record).
    * ``client`` — a polite HTTP client used to fetch external writeup bodies. If
      None, writeups with only a URL are recorded but not fetched.

    Mutates and returns ``challenge`` with ``corpus_path`` set, writeup ``local_path``
    populated, and writeup files added to the file manifest.
    """
    dest = corpus_root / challenge_relpath(challenge)
    dest.mkdir(parents=True, exist_ok=True)

    if raw_dir is not None and Path(raw_dir).exists():
        _copy_sources(Path(raw_dir), dest)

    # Materialize writeups as hard copies under <challenge>/writeups/.
    wdir = dest / "writeups"
    if challenge.writeups:
        wdir.mkdir(exist_ok=True)
        streamer = _stream_client(client) if client is not None else None
        kept: set[str] = set()
        for i, wu in enumerate(challenge.writeups):
            content: bytes | None = None
            ext = ".md"
            if wu.text:  # body already available inline at the source
                content = wu.text.encode("utf-8")
                ext = ".md"
            elif wu.url and streamer is not None:  # follow the external link, grab bytes
                content, ext = _fetch_writeup(streamer, client, str(wu.url))

            if content is None:
                continue
            fname = f"writeup_{i:02d}{ext}"
            (wdir / fname).write_bytes(content)
            kept.add(fname)
            rel = f"writeups/{fname}"
            wu.local_path = rel
            wu.text = None  # bytes now live on disk; keep the catalog lean
            challenge.files.append(
                FileEntry(
                    path=rel,
                    sha256=sha256_bytes(content),
                    size=len(content),
                    is_source=is_source_file(rel),
                )
            )
        _prune_writeups(wdir, kept)
    elif wdir.exists():
        # No writeups this run: drop any stale hard copies from a previous ingest.
        _prune_writeups(wdir, set())

    # Record corpus_path relative to the repo root when known (portable in the
    # catalog). Both sides are resolved so an absolute --data-dir still yields a
    # repo-relative path (e.g. 'corpus/github/...') rather than a machine path.
    if repo_root is not None:
        try:
            challenge.corpus_path = str(dest.resolve().relative_to(Path(repo_root).resolve()))
        except ValueError:
            challenge.corpus_path = str(dest)
    else:
        challenge.corpus_path = str(dest)
    return challenge


def _fetch_writeup(streamer, client: PoliteClient, url: str) -> tuple[bytes | None, str]:
    """Fetch one external writeup body through the SSRF-safe, byte-capped path.

    Rate-limits via the wrapping ``PoliteClient``'s limiter when present, then
    delegates to :func:`ctfhoard.netguard.safe_get`. Unsafe URLs (SSRF targets,
    over-cap bodies) and transient fetch errors are logged and skipped — one bad
    link must never abort the challenge. Returns ``(content_or_None, ext)``.
    """
    limiter = getattr(client, "_limiter", None)
    if limiter is not None:
        limiter.acquire()
    try:
        status, body, content_type = safe_get(streamer, url)
    except UnsafeUrlError as exc:
        logger.debug("skipping unsafe writeup url {}: {}", url, exc)
        return None, ".md"
    except Exception as exc:  # noqa: BLE001 — one bad link must not abort the challenge
        logger.debug("writeup fetch failed for {}: {}", url, exc)
        return None, ".md"
    if status == 200 and body:
        return body, _writeup_ext(url, content_type)
    return None, ".md"


def _prune_writeups(wdir: Path, kept: set[str]) -> None:
    """Remove writeup hard copies not (re)written this run, so corpus == catalog.

    Re-ingestion can leave a stale ``writeup_01.html`` that no catalog entry
    references (fewer/renamed writeups than before); drop such orphans.
    """
    for existing in wdir.iterdir():
        if existing.is_file() and existing.name not in kept:
            existing.unlink()


def archive_challenge(challenge_dir: Path) -> Path:
    """Pack one materialized challenge dir into a single deterministic ``.tar.gz``.

    Hugging Face rate-limits per-file uploads (~5-11 files/s regardless of bandwidth or
    Xet/LFS), so publishing the corpus one loose source file at a time is infeasible at
    the scale of millions of tiny files. Collapsing EACH challenge into one tarball cuts
    the file count ~1000× (millions of sources → tens of thousands of per-challenge
    archives), which is tractable. Each archive is self-contained and independently
    extractable — its members are the challenge dir's contents at paths *relative* to the
    challenge dir (writeups included, since they already live under it), so extracting
    reproduces the challenge tree.

    Determinism: members are added in sorted order with a fixed ``mtime=0`` and zeroed
    owner metadata, and gzip is written with ``mtime=0`` (no embedded timestamp/filename),
    so re-archiving byte-identical content yields a byte-identical tarball. HF then diffs
    it away and skips re-uploading unchanged challenges.

    The loose challenge dir is removed after archiving, leaving only ``<dir>.tar.gz`` in
    staging. Returns the archive path.
    """
    challenge_dir = Path(challenge_dir)
    archive_path = challenge_dir.with_name(challenge_dir.name + ".tar.gz")

    # Gather regular files only, at paths relative to the challenge dir, sorted for a
    # stable member order. os.walk does not follow symlinks; _copy_sources already
    # refused symlinked sources, so the tree is plain files/dirs.
    members: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(challenge_dir):
        dirnames.sort()
        for fn in sorted(filenames):
            full = Path(dirpath) / fn
            arcname = full.relative_to(challenge_dir).as_posix()
            members.append((full, arcname))
    members.sort(key=lambda t: t[1])

    def _reset(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.mtime = 0
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        return ti

    # GNU format (not USTAR): USTAR caps member names at 100 chars, and real repos
    # (e.g. google-ctf) have deeper/longer paths that raise "name is too long" and fail
    # the whole repo. GNU handles arbitrary-length names and stays byte-deterministic
    # with a fixed mtime (unlike PAX, which embeds atime/ctime headers).
    # gzip with mtime=0 and no fileobj filename → byte-stable header across re-runs.
    with (
        open(archive_path, "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar,
    ):
        for full, arcname in members:
            tar.add(full, arcname=arcname, recursive=False, filter=_reset)

    shutil.rmtree(challenge_dir)
    return archive_path


def materialize_and_archive(
    challenge: Challenge,
    corpus_root: Path,
    *,
    raw_dir: Path | None = None,
    client: PoliteClient | None = None,
    repo_root: Path | None = None,
) -> Challenge:
    """Materialize a challenge, then pack its dir into one per-challenge ``.tar.gz``.

    Layers per-challenge archiving on top of :func:`materialize_challenge`: the loose
    challenge tree is written first (so the browsable per-file ``files`` manifest is still
    computed exactly as before), then the whole dir is replaced by a single deterministic
    ``<slug>__id.tar.gz`` via :func:`archive_challenge`. Only ``corpus_path`` changes — it
    is repointed at the archive (e.g. ``corpus/<origin>/<event>/<year>/<slug>__id.tar.gz``,
    repo-relative when ``repo_root`` is known); the per-file ``files`` index is preserved.

    This is what makes the HF publish tractable: a batch of N repos then stages roughly
    one tarball per challenge instead of the challenges' millions of loose source files,
    and each archive remains independently extractable.
    """
    materialize_challenge(
        challenge, corpus_root, raw_dir=raw_dir, client=client, repo_root=repo_root
    )
    dest = corpus_root / challenge_relpath(challenge)
    archive = archive_challenge(dest)
    if repo_root is not None:
        try:
            challenge.corpus_path = str(
                archive.resolve().relative_to(Path(repo_root).resolve())
            )
        except ValueError:
            challenge.corpus_path = str(archive)
    else:
        challenge.corpus_path = str(archive)
    return challenge
