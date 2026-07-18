"""Normalized data contract for the CTF corpus.

Every connector, whatever its source (CTFtime, a GitHub repo, Hackropole,
Juice Shop...), must ultimately emit :class:`Challenge` records that conform to
these models. This is the single schema the whole pipeline agrees on: connectors
produce it, the deduper merges it, the catalog serializes it (JSONL/Parquet), and
consumers (humans or agents) read it.

Design notes
------------
* Provenance and licensing are first-class, per challenge. A "mirror everything"
  corpus aggregates code under heterogeneous (and often absent) licenses, so each
  record carries enough attribution + SPDX detection to later filter what is
  actually redistributable (see :mod:`ctfhoard.licenses`).
* ``content_fingerprint`` lets the deduper recognize the same challenge appearing
  in many repos (e.g. an official event repo, sajjadium/ctf-archives, and a team
  fork) and collapse them into one canonical record with an attribution graph.
* Models are intentionally permissive on optional metadata: real-world sources are
  inconsistent, and we would rather ingest a partial record than drop it.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class Category(enum.StrEnum):
    """Coarse challenge category, normalized across sources.

    Sources use wildly different labels ("Web Exploitation", "web", "Web/PHP").
    Connectors map their raw label to one of these; the raw label is preserved in
    :attr:`Challenge.raw_category` for traceability.
    """

    PWN = "pwn"
    WEB = "web"
    REVERSE = "reverse"
    CRYPTO = "crypto"
    FORENSICS = "forensics"
    STEGANOGRAPHY = "stego"
    MISC = "misc"
    HARDWARE = "hardware"
    OSINT = "osint"
    MOBILE = "mobile"
    BLOCKCHAIN = "blockchain"
    NETWORKING = "networking"
    PROGRAMMING = "programming"
    UNKNOWN = "unknown"


class Origin(enum.StrEnum):
    """Where a piece of data was retrieved from (the connector's identity)."""

    CTFTIME = "ctftime"
    GITHUB = "github"
    HACKROPOLE = "hackropole"
    PICOCTF = "picoctf"
    JUICESHOP = "juiceshop"
    NYU_CTF_BENCH = "nyu_ctf_bench"
    SAJJADIUM = "sajjadium"
    PWNCOLLEGE = "pwncollege"
    GOOGLE_CTF = "google_ctf"
    OTHER = "other"


class LicenseInfo(BaseModel):
    """Detected licensing for a source, driving redistribution decisions."""

    spdx_id: str | None = Field(
        default=None,
        description="Detected SPDX identifier (e.g. 'MIT', 'Apache-2.0', "
        "'etalab-2.0'). None when no license could be detected.",
    )
    name: str | None = Field(default=None, description="Human-readable license name.")
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0..1 confidence of the detection (1.0 = explicit SPDX from a "
        "trusted API such as GitHub's license endpoint).",
    )
    redistributable: bool = Field(
        default=False,
        description="Derived flag: may this content be redistributed in an open "
        "aggregate? Conservative — False when no license is detected "
        "(all-rights-reserved by default).",
    )
    source_file: str | None = Field(
        default=None,
        description="Path of the LICENSE/COPYING file the detection came from, if any.",
    )
    note: str | None = Field(
        default=None,
        description="Free-form caveat, e.g. 'aggregation MIT but per-challenge "
        "author copyright applies'.",
    )


class FileEntry(BaseModel):
    """One artifact belonging to a challenge (handout binary, Dockerfile, source...)."""

    path: str = Field(description="Path relative to the challenge root.")
    sha256: str = Field(description="Hex SHA-256 of the file content.")
    size: int = Field(ge=0, description="Size in bytes.")
    is_source: bool = Field(
        default=False,
        description="True if this file is challenge *source* (the whitebox part), "
        "as opposed to a writeup/solve/README.",
    )
    lfs: bool = Field(
        default=False,
        description="True if stored as a Git LFS pointer rather than inline "
        "(large binaries/pcaps/disk images over the size cap).",
    )
    lfs_oid: str | None = Field(
        default=None, description="Expected LFS object id when the blob is not mirrored."
    )
    media_type: str | None = Field(default=None, description="Guessed MIME type.")


class Source(BaseModel):
    """A single provenance edge: where one copy of the challenge came from.

    A deduplicated challenge may carry several sources (the same challenge mirrored
    from an official repo, a community archive, and a team fork). The first source
    is the canonical one chosen by the deduper.
    """

    origin: Origin
    url: HttpUrl | None = Field(default=None, description="Canonical URL of this copy.")
    repo: str | None = Field(
        default=None, description="'owner/name' when the source is a GitHub repo."
    )
    commit_sha: str | None = Field(
        default=None, description="Exact commit the content was pinned/mirrored at."
    )
    path_in_repo: str | None = Field(
        default=None, description="Sub-path of the challenge inside the repo."
    )
    license: LicenseInfo = Field(default_factory=LicenseInfo)
    retrieved_at: datetime | None = Field(
        default=None, description="When this copy was fetched (UTC)."
    )
    is_official: bool = Field(
        default=False,
        description="True when this is a first-party organizer source (highest "
        "authority for canonicalization).",
    )


class Writeup(BaseModel):
    """A solution writeup attached to a challenge."""

    url: HttpUrl | None = Field(default=None)
    origin: Origin = Origin.OTHER
    author: str | None = Field(default=None, description="Author or team name.")
    title: str | None = Field(default=None)
    text: str | None = Field(
        default=None,
        description="Full writeup text when inline/scrapable. None when we only hold "
        "a reference link (e.g. CTFtime 'ai-train=no' content kept as reference).",
    )
    is_inline: bool = Field(
        default=False,
        description="True if `text` holds the writeup body; False if this is a "
        "reference link only.",
    )
    rating: float | None = Field(default=None, description="Community rating if any.")
    language: str | None = Field(default=None, description="Natural language (ISO 639-1).")
    retrieved_at: datetime | None = Field(default=None)


class CtfEvent(BaseModel):
    """A CTF edition (one year's instance of a series), from CTFtime metadata."""

    ctftime_event_id: int | None = Field(
        default=None, description="CTFtime event id (the yearly edition)."
    )
    ctftime_series_id: int | None = Field(
        default=None,
        description="CTFtime `ctf_id` — stable across editions; group editions by this.",
    )
    name: str = Field(description="Series/event title, e.g. 'HITCON CTF'.")
    edition: str | None = Field(default=None, description="Edition label, e.g. 'Quals'.")
    year: int | None = Field(default=None)
    start: datetime | None = Field(default=None)
    finish: datetime | None = Field(default=None)
    weight: float | None = Field(default=None, description="CTFtime difficulty weight.")
    format: str | None = Field(default=None, description="'Jeopardy', 'Attack-Defense'.")
    ctftime_url: HttpUrl | None = Field(default=None)
    homepage: HttpUrl | None = Field(default=None, description="Organizer's own site.")


class Challenge(BaseModel):
    """The central record: one whitebox CTF challenge, normalized and deduplicated."""

    # --- Identity -----------------------------------------------------------
    id: str = Field(
        description="Stable synthetic id: hash of (event slug, year, normalized "
        "title, category). Survives re-ingestion so records can be upserted."
    )
    title: str
    slug: str | None = Field(default=None, description="URL/filesystem-safe title.")

    # --- Event context ------------------------------------------------------
    event: CtfEvent | None = Field(default=None)
    event_name: str | None = Field(
        default=None, description="Denormalized event name for cheap filtering/search."
    )
    year: int | None = Field(default=None)

    # --- Classification -----------------------------------------------------
    category: Category = Category.UNKNOWN
    raw_category: str | None = Field(
        default=None, description="The source's original category label, verbatim."
    )
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = Field(
        default=None, description="Normalized difficulty label if known."
    )
    points: int | None = Field(default=None)

    # --- Statement / content ------------------------------------------------
    description: str | None = Field(
        default=None, description="Challenge statement/prompt (may contain HTML/MD)."
    )
    flag: str | None = Field(
        default=None, description="The flag, when published in the source."
    )
    flag_format: str | None = Field(
        default=None, description="e.g. 'flag{...}', 'FCSC{...}'."
    )

    # --- Whitebox artifacts -------------------------------------------------
    has_source: bool = Field(
        default=False,
        description="True when challenge SOURCE (not just a handout binary) is present. "
        "This is the primary filter for a *whitebox* corpus.",
    )
    files: list[FileEntry] = Field(default_factory=list)
    solve_languages: list[str] = Field(
        default_factory=list, description="Detected languages of solve scripts/sources."
    )
    docker: bool = Field(
        default=False, description="True if a Dockerfile/compose ships with the challenge."
    )

    # --- Writeups -----------------------------------------------------------
    writeups: list[Writeup] = Field(default_factory=list)

    # --- Provenance & licensing --------------------------------------------
    sources: list[Source] = Field(
        default_factory=list,
        description="Provenance edges. sources[0] is canonical after dedup.",
    )
    license: LicenseInfo = Field(
        default_factory=LicenseInfo,
        description="Effective license for the canonical copy (drives redistribution).",
    )

    # --- Dedup --------------------------------------------------------------
    content_fingerprint: str | None = Field(
        default=None,
        description="Merkle hash over the normalized source-file set (README/writeups "
        "excluded) — identical across mirrors of the same challenge.",
    )
    duplicate_of: str | None = Field(
        default=None,
        description="If this record was merged into another, the canonical Challenge.id.",
    )

    # --- Bookkeeping --------------------------------------------------------
    corpus_path: str | None = Field(
        default=None, description="Path of the mirrored challenge dir under data/corpus/."
    )
    ingested_at: datetime | None = Field(default=None)
    schema_version: int = Field(default=1)

    @property
    def redistributable(self) -> bool:
        """Whether the canonical copy may be redistributed in the open aggregate."""
        return self.license.redistributable


class RawChallenge(BaseModel):
    """Loose intermediate a connector yields before normalization.

    Connectors do the source-specific extraction and hand back this permissive
    shape; :mod:`ctfhoard.normalize` turns it into a canonical :class:`Challenge`
    (id synthesis, category mapping, fingerprinting, license resolution). Keeping
    connectors decoupled from the strict contract makes them small and testable.
    """

    origin: Origin
    title: str
    event_name: str | None = None
    edition: str | None = None
    year: int | None = None
    raw_category: str | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    points: int | None = None
    description: str | None = None
    flag: str | None = None
    # Local directory holding the challenge artifacts, if the connector materialized
    # them on disk (git/tarball mirror). normalize() walks it to build FileEntry list.
    local_dir: str | None = None
    files: list[FileEntry] = Field(default_factory=list)
    writeups: list[Writeup] = Field(default_factory=list)
    source: Source | None = None
    extra: dict = Field(
        default_factory=dict, description="Connector-specific passthrough metadata."
    )
