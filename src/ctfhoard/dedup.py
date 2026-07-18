"""Content-addressed deduplication.

The same challenge is mirrored in many places: the official event repo,
sajjadium/ctf-archives, pwn.college's archive, individual team forks. We collapse
these into one canonical :class:`~ctfhoard.schema.Challenge` with an attribution
graph (all provenance edges preserved) instead of storing N near-copies.

Two levels:

* **File-level** — SHA-256 of every file. The *distributed artifact* (handout
  binary, pcap) hash-matches exactly across repos even when the surrounding
  writeup/solve differ.
* **Challenge-unit level** — a Merkle-style fingerprint over the *source* file set
  only (README/writeup/solve/.git excluded), so a "source" copy and a
  "source+writeup" copy of the same challenge still fingerprint identically.

Near-duplicate (reformatted / renamed) detection via MinHash is optional and lives
behind the ``dedup`` extra; :func:`minhash_signature` degrades gracefully when
``datasketch`` is absent.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from ctfhoard.schema import Challenge, FileEntry, Source

# Files that are NOT challenge source — excluded from the content fingerprint so
# that mirrors carrying extra writeups still collapse onto the same challenge.
_NON_SOURCE_NAMES = re.compile(
    r"""^(
        readme(\..*)?         |
        writeup(s)?(\..*)?    |
        solve(r|s)?(\..*)?    |
        solution(s)?(\..*)?   |
        \.git.*               |
        \.ds_store
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_source_file(relative_path: str) -> bool:
    """Heuristic: does this path count as challenge *source* (whitebox) content?"""
    name = Path(relative_path).name
    return not _NON_SOURCE_NAMES.match(name)


def content_fingerprint(files: Iterable[FileEntry]) -> str | None:
    """Merkle-style fingerprint over the normalized source-file set.

    Deterministic and order-independent: we sort ``(path, sha256)`` tuples of the
    source files, join, and hash. Two challenge directories with identical source
    (regardless of extra writeups/READMEs, or file discovery order) yield the same
    fingerprint. Returns None when there are no source files (nothing to anchor on).

    Every source file participates. We anchor on its ``sha256``; if that is somehow
    missing we fall back to ``lfs_oid`` and finally to ``lfs:<size>`` so a large
    source file is never silently dropped (which would let two challenges differing
    only in that file collide onto one fingerprint and be wrongly merged).
    """
    entries = sorted(
        (f.path, f.sha256 or f.lfs_oid or f"lfs:{f.size}") for f in files if f.is_source
    )
    if not entries:
        return None
    joined = "\n".join(f"{p}:{h}" for p, h in entries)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _authority(source: Source) -> tuple[int, int]:
    """Sort key for choosing the canonical source of a challenge cluster.

    Higher is better: first-party official repos beat community archives beat forks.
    Ties broken later by caller (e.g. earliest retrieval).
    """
    from ctfhoard.schema import Origin

    official = 1 if source.is_official else 0
    origin_rank = {
        Origin.GOOGLE_CTF: 5,
        Origin.HACKROPOLE: 5,
        Origin.JUICESHOP: 5,
        Origin.NYU_CTF_BENCH: 4,
        Origin.SAJJADIUM: 3,
        Origin.PWNCOLLEGE: 3,
        Origin.CTFTIME: 2,
        Origin.GITHUB: 1,
        Origin.OTHER: 0,
    }.get(source.origin, 0)
    return (official, origin_rank)


def merge_cluster(members: list[Challenge]) -> Challenge:
    """Merge challenges sharing a fingerprint into one canonical record.

    The member with the most authoritative source wins as the base; every other
    member's sources and writeups are folded in as provenance/attribution. The
    losing members should be marked ``duplicate_of`` the canonical id by the caller.
    """
    if not members:
        raise ValueError("cannot merge an empty cluster")
    if len(members) == 1:
        return members[0]

    def best_source_rank(ch: Challenge) -> tuple[int, int]:
        return max((_authority(s) for s in ch.sources), default=(0, 0))

    canonical = max(members, key=best_source_rank)

    seen_sources = {(s.origin, str(s.url), s.repo, s.path_in_repo) for s in canonical.sources}
    seen_writeups = {(w.origin, str(w.url)) for w in canonical.writeups}

    for member in members:
        if member is canonical:
            continue
        for s in member.sources:
            key = (s.origin, str(s.url), s.repo, s.path_in_repo)
            if key not in seen_sources:
                seen_sources.add(key)
                canonical.sources.append(s)
        for w in member.writeups:
            key = (w.origin, str(w.url))
            if key not in seen_writeups:
                seen_writeups.add(key)
                canonical.writeups.append(w)

    # Re-elect the effective license from the canonical (first) source.
    if canonical.sources:
        canonical.sources.sort(key=_authority, reverse=True)
        canonical.license = canonical.sources[0].license
    return canonical


def cluster_by_fingerprint(challenges: Iterable[Challenge]) -> dict[str, list[Challenge]]:
    """Group challenges by ``content_fingerprint``.

    Records without a fingerprint (no source files) are returned each in their own
    singleton cluster keyed by their id — they cannot be safely merged on content.
    """
    clusters: dict[str, list[Challenge]] = {}
    for ch in challenges:
        key = ch.content_fingerprint or f"__nofp__:{ch.id}"
        clusters.setdefault(key, []).append(ch)
    return clusters


def minhash_signature(tokens: Iterable[str], num_perm: int = 128):
    """MinHash signature for near-duplicate text detection (optional).

    Returns a ``datasketch.MinHash`` when the ``dedup`` extra is installed, else
    None so callers can skip near-dup detection without a hard dependency.
    """
    try:
        from datasketch import MinHash
    except ImportError:
        return None
    mh = MinHash(num_perm=num_perm)
    for tok in tokens:
        mh.update(tok.encode("utf-8"))
    return mh


def normalize_tokens(text: str) -> list[str]:
    """Whitespace/comment-insensitive token stream for near-dup hashing."""
    stripped = re.sub(r"(#|//).*?$", "", text, flags=re.MULTILINE)
    return re.findall(r"[A-Za-z0-9_]+", stripped.lower())
