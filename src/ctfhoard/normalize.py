"""Turn a connector's loose ``RawChallenge`` into a canonical ``Challenge``.

This is the one place where source-specific messiness becomes uniform: stable id
synthesis, category mapping, file-manifest construction (walking ``local_dir`` when
the connector mirrored files), source-language detection, and content
fingerprinting. Connectors stay dumb; all the contract-level logic lives here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from ctfhoard.dedup import content_fingerprint, is_source_file, sha256_file
from ctfhoard.schema import Category, Challenge, CtfEvent, FileEntry, RawChallenge

# Raw category label (lowercased, stripped) → normalized Category. Substring match,
# longest-key-first, so "web exploitation" maps before a bare "web" fallback.
_CATEGORY_MAP: dict[str, Category] = {
    "pwn": Category.PWN,
    "binary exploitation": Category.PWN,
    "binary": Category.PWN,
    "exploitation": Category.PWN,
    "web exploitation": Category.WEB,
    "web": Category.WEB,
    "reverse engineering": Category.REVERSE,
    "reversing": Category.REVERSE,
    "reverse": Category.REVERSE,
    "rev": Category.REVERSE,
    "cryptography": Category.CRYPTO,
    "crypto": Category.CRYPTO,
    "forensics": Category.FORENSICS,
    "forensic": Category.FORENSICS,
    "steganography": Category.STEGANOGRAPHY,
    "stego": Category.STEGANOGRAPHY,
    "hardware": Category.HARDWARE,
    "osint": Category.OSINT,
    "mobile": Category.MOBILE,
    "android": Category.MOBILE,
    "blockchain": Category.BLOCKCHAIN,
    "smart contract": Category.BLOCKCHAIN,
    "networking": Category.NETWORKING,
    "network": Category.NETWORKING,
    "programming": Category.PROGRAMMING,
    "ppc": Category.PROGRAMMING,
    "misc": Category.MISC,
    "miscellaneous": Category.MISC,
    # Web-application vulnerability classes (OWASP-style labels, e.g. Juice Shop):
    # these are all web challenges unless they clearly belong elsewhere.
    "injection": Category.WEB,
    "xss": Category.WEB,
    "cross site scripting": Category.WEB,
    "broken access control": Category.WEB,
    "broken authentication": Category.WEB,
    "sensitive data exposure": Category.WEB,
    "security misconfiguration": Category.WEB,
    "vulnerable components": Category.WEB,
    "improper input validation": Category.WEB,
    "insecure deserialization": Category.WEB,
    "xxe": Category.WEB,
    "server side request forgery": Category.WEB,
    "ssrf": Category.WEB,
    "unvalidated redirects": Category.WEB,
    "forgotten content": Category.WEB,
    "security through obscurity": Category.MISC,
    "cryptographic issues": Category.CRYPTO,
}

_SOURCE_LANG_EXT = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".js": "javascript",
    ".ts": "typescript",
    ".php": "php",
    ".rb": "ruby",
    ".java": "java",
    ".sol": "solidity",
    ".asm": "asm",
    ".s": "asm",
    ".sh": "shell",
}

_DOCKER_NAMES = {"dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yaml"}
# Unicode-aware: collapse runs of non-word chars (and underscores) into a single
# separator. ``\w`` under ``re.UNICODE`` keeps CJK/Cyrillic/etc. word characters, so
# non-ASCII titles produce a non-empty slug instead of collapsing to ''. ASCII
# behavior is unchanged ('HITCON CTF' -> 'hitcon-ctf', 'baby_pwn' -> 'baby-pwn').
_SLUG_RE = re.compile(r"[\W_]+", re.UNICODE)


def slugify(text: str) -> str:
    """Lower-cased, Unicode-aware URL/filesystem slug.

    NFKC-normalizes first so compatibility variants fold together, then replaces
    runs of non-word characters with '-'. Returns '' only for input with no word
    characters at all (e.g. emoji/punctuation-only); callers must guard against that.
    """
    normalized = unicodedata.normalize("NFKC", text)
    return _SLUG_RE.sub("-", normalized.lower()).strip("-")


def map_category(raw: str | None) -> Category:
    """Map a source's raw category label to the normalized enum."""
    if not raw:
        return Category.UNKNOWN
    needle = raw.strip().lower()
    for key in sorted(_CATEGORY_MAP, key=len, reverse=True):
        if key in needle:
            return _CATEGORY_MAP[key]
    return Category.UNKNOWN


def synthesize_id(
    event_name: str | None, year: int | None, title: str, category: str | None
) -> str:
    """Stable synthetic id: hash of (event, year, title, category).

    Deterministic across re-ingestion so records upsert instead of duplicating.

    The title contributes its slug, but a slug can collapse to '' (emoji/punctuation
    -only titles). In that case distinct titles would share an id and last-write-wins
    merging would silently drop challenges, so we anchor on a stable hash of the RAW
    title instead — guaranteeing distinct titles always yield distinct ids.
    """
    title_slug = slugify(title)
    title_basis = title_slug or f"h:{hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]}"
    basis = "|".join(
        [
            slugify(event_name or "unknown-event"),
            str(year or "0000"),
            title_basis,
            (category or "unknown").lower(),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def build_file_manifest(local_dir: Path, *, size_cap: int = 50 * 1024 * 1024) -> list[FileEntry]:
    """Walk a mirrored challenge dir into a list of ``FileEntry``.

    Files above ``size_cap`` are flagged ``lfs=True`` (their blob should be handled
    by Git LFS / recorded as a pointer, not inlined into the catalog). Hashing is
    decoupled from that flag: we ALWAYS compute the streaming SHA-256 (cheap on
    memory), so a large source file still carries a stable content identity. Leaving
    ``sha256`` empty for big files would let two challenges that differ only in a
    large binary share a content fingerprint and be wrongly merged.
    """
    entries: list[FileEntry] = []
    if not local_dir.exists():
        return entries
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(local_dir).as_posix()
        if "/.git/" in f"/{rel}" or rel.startswith(".git/"):
            continue
        size = path.stat().st_size
        big = size > size_cap
        entries.append(
            FileEntry(
                path=rel,
                sha256=sha256_file(path),
                size=size,
                is_source=is_source_file(rel),
                lfs=big,
                media_type=None,
            )
        )
    return entries


def _detect_languages(files: list[FileEntry]) -> list[str]:
    langs: set[str] = set()
    for f in files:
        if not f.is_source:
            continue
        lang = _SOURCE_LANG_EXT.get(Path(f.path).suffix.lower())
        if lang:
            langs.add(lang)
    return sorted(langs)


def _has_docker(files: list[FileEntry]) -> bool:
    return any(Path(f.path).name.lower() in _DOCKER_NAMES for f in files)


def normalize(raw: RawChallenge) -> Challenge:
    """Build a canonical :class:`Challenge` from a connector's ``RawChallenge``."""
    files = list(raw.files)
    if raw.local_dir:
        files.extend(build_file_manifest(Path(raw.local_dir)))

    category = map_category(raw.raw_category)
    challenge_id = synthesize_id(raw.event_name, raw.year, raw.title, category.value)
    # Guard against an empty slug (emoji/punctuation-only titles) by falling back to
    # the id prefix, so ``slug`` is always usable as a URL/filesystem key.
    slug = slugify(raw.title) or f"chal-{challenge_id[:8]}"

    event = None
    if raw.event_name:
        event = CtfEvent(name=raw.event_name, edition=raw.edition, year=raw.year)

    sources = [raw.source] if raw.source else []
    license_info = raw.source.license if raw.source else None

    has_source = any(f.is_source for f in files)

    challenge = Challenge(
        id=challenge_id,
        title=raw.title,
        slug=slug,
        event=event,
        event_name=raw.event_name,
        year=raw.year,
        category=category,
        raw_category=raw.raw_category,
        tags=raw.tags,
        difficulty=raw.difficulty,
        points=raw.points,
        description=raw.description,
        flag=raw.flag,
        has_source=has_source,
        files=files,
        solve_languages=_detect_languages(files),
        docker=_has_docker(files),
        writeups=raw.writeups,
        sources=sources,
        license=license_info or Challenge.model_fields["license"].default_factory(),
        content_fingerprint=content_fingerprint(files),
        corpus_path=raw.local_dir,
        ingested_at=datetime.now(UTC),
    )
    return challenge
