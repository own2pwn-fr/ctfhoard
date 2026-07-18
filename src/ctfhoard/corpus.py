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
    """Copy a connector's downloaded challenge tree into the corpus, minus VCS noise."""
    for src in raw_dir.rglob("*"):
        if src.is_dir() or src.is_symlink():
            continue
        rel = src.relative_to(raw_dir)
        if ".git" in rel.parts:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


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
    if challenge.writeups:
        wdir = dest / "writeups"
        wdir.mkdir(exist_ok=True)
        for i, wu in enumerate(challenge.writeups):
            content: bytes | None = None
            ext = ".md"
            if wu.text:  # body already available inline at the source
                content = wu.text.encode("utf-8")
                ext = ".md"
            elif wu.url and client is not None:  # follow the external link, grab bytes
                try:
                    resp = client.get(str(wu.url))
                    if resp.status_code == 200 and resp.content:
                        content = resp.content
                        ext = _writeup_ext(str(wu.url), resp.headers.get("content-type"))
                except Exception as exc:  # noqa: BLE001 — one bad link must not abort
                    logger.debug("writeup fetch failed for {}: {}", wu.url, exc)

            if content is None:
                continue
            fname = f"writeup_{i:02d}{ext}"
            (wdir / fname).write_bytes(content)
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

    # Record corpus_path relative to the repo root when known (portable in the catalog).
    challenge.corpus_path = (
        str(dest.relative_to(repo_root)) if repo_root else str(dest)
    )
    return challenge
