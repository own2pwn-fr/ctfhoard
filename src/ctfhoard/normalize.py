"""Turn a connector's loose ``RawChallenge`` into a canonical ``Challenge``.

This is the one place where source-specific messiness becomes uniform: stable id
synthesis, category mapping, file-manifest construction (walking ``local_dir`` when
the connector mirrored files), source-language detection, and content
fingerprinting. Connectors stay dumb; all the contract-level logic lives here.
"""

from __future__ import annotations

import hashlib
import re
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
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


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
    """
    basis = "|".join(
        [
            slugify(event_name or "unknown-event"),
            str(year or "0000"),
            slugify(title),
            (category or "unknown").lower(),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def build_file_manifest(local_dir: Path, *, size_cap: int = 50 * 1024 * 1024) -> list[FileEntry]:
    """Walk a mirrored challenge dir into a list of ``FileEntry``.

    Files above ``size_cap`` are flagged ``lfs=True`` (their blob should be handled
    by Git LFS / recorded as a pointer, not inlined into the catalog).
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
                sha256="" if big else sha256_file(path),
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

    event = None
    if raw.event_name:
        event = CtfEvent(name=raw.event_name, edition=raw.edition, year=raw.year)

    sources = [raw.source] if raw.source else []
    license_info = raw.source.license if raw.source else None

    has_source = any(f.is_source for f in files)

    challenge = Challenge(
        id=challenge_id,
        title=raw.title,
        slug=slugify(raw.title),
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
