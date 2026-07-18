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

import re
import shutil
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
    for src in raw_dir.rglob("*"):
        if src.is_dir() or src.is_symlink():
            continue
        # Refuse anything that resolves outside raw_dir (symlinked-directory escape).
        try:
            if not src.resolve().is_relative_to(root):
                continue
        except OSError:
            continue
        rel = src.relative_to(raw_dir)
        if ".git" in rel.parts:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


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
