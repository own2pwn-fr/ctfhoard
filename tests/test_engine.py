"""Tests for the source-independent engine: licenses, normalize, dedup, ratelimit.

These lock the data contract every connector relies on, so a connector change (or a
new connector) can't silently break id synthesis, fingerprinting, or mirror merging.
"""

from __future__ import annotations

import time

from ctfhoard.dedup import (
    cluster_by_fingerprint,
    content_fingerprint,
    is_source_file,
    merge_cluster,
)
from ctfhoard.licenses import detect_from_text, from_spdx, is_redistributable
from ctfhoard.normalize import map_category, normalize, synthesize_id
from ctfhoard.ratelimit import RateLimiter
from ctfhoard.schema import (
    Category,
    FileEntry,
    LicenseInfo,
    Origin,
    RawChallenge,
    Source,
    Writeup,
)

# --------------------------------------------------------------------------- #
# licenses
# --------------------------------------------------------------------------- #


def test_redistributable_decision():
    assert is_redistributable("MIT")
    assert is_redistributable("etalab-2.0")  # Hackropole content
    assert is_redistributable("GPL-2.0-or-later")
    assert not is_redistributable("NOASSERTION")
    assert not is_redistributable(None)  # no license == all-rights-reserved


def test_from_spdx_flags():
    mit = from_spdx("MIT")
    assert mit.spdx_id == "MIT" and mit.redistributable and mit.confidence == 1.0
    none = from_spdx("NOASSERTION")
    assert none.spdx_id is None and not none.redistributable


def test_detect_from_text_mit_and_unknown():
    mit_text = "Permission is hereby granted, free of charge, to any person obtaining a copy"
    info = detect_from_text(mit_text, source_file="LICENSE")
    assert info.spdx_id == "MIT" and info.redistributable and info.source_file == "LICENSE"

    unknown = detect_from_text("All rights reserved. Proprietary.")
    assert unknown.spdx_id is None and not unknown.redistributable


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #


def test_category_mapping_prefers_specific():
    assert map_category("Web Exploitation") is Category.WEB
    assert map_category("binary exploitation") is Category.PWN
    assert map_category("Reverse Engineering") is Category.REVERSE
    assert map_category("totally unknown thing") is Category.UNKNOWN
    assert map_category(None) is Category.UNKNOWN


def test_synthesize_id_is_stable_and_slug_insensitive():
    a = synthesize_id("HITCON CTF", 2021, "Baby Pwn", "pwn")
    b = synthesize_id("hitcon-ctf", 2021, "baby-pwn", "pwn")
    assert a == b  # slugified basis → same id regardless of surface formatting
    assert a != synthesize_id("HITCON CTF", 2022, "Baby Pwn", "pwn")


def _raw(title="SQLi", cat="Web", files=None, official=True, spdx="MIT"):
    return RawChallenge(
        origin=Origin.JUICESHOP,
        title=title,
        event_name="Juice Shop",
        year=2024,
        raw_category=cat,
        files=files or [],
        source=Source(
            origin=Origin.JUICESHOP,
            is_official=official,
            license=from_spdx(spdx) if spdx else LicenseInfo(),
        ),
    )


def test_normalize_builds_whitebox_and_fingerprint():
    raw = _raw(files=[FileEntry(path="chal.py", sha256="a" * 64, size=12, is_source=True)])
    ch = normalize(raw)
    assert ch.category is Category.WEB
    assert ch.has_source
    assert ch.content_fingerprint  # source present → fingerprint anchored
    assert ch.redistributable
    assert "python" in ch.solve_languages


def test_normalize_no_source_has_no_fingerprint():
    ch = normalize(_raw(files=[FileEntry(path="README.md", sha256="b" * 64, size=3)]))
    assert not ch.has_source
    assert ch.content_fingerprint is None  # nothing to anchor dedup on


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #


def test_is_source_file_excludes_writeups():
    assert is_source_file("src/chal.c")
    assert not is_source_file("README.md")
    assert not is_source_file("writeup.md")
    assert not is_source_file("solve.py")


def test_fingerprint_ignores_writeups_and_order():
    src = FileEntry(path="chal.py", sha256="a" * 64, size=1, is_source=True)
    wu = FileEntry(path="writeup.md", sha256="c" * 64, size=1, is_source=False)
    fp_source_only = content_fingerprint([src])
    fp_with_writeup = content_fingerprint([wu, src])  # extra writeup + different order
    assert fp_source_only == fp_with_writeup  # mirrors collapse onto same challenge


def test_merge_cluster_prefers_official_and_folds_provenance():
    src = [FileEntry(path="chal.py", sha256="a" * 64, size=1, is_source=True)]
    official = normalize(_raw(official=True, files=src, spdx="MIT"))
    official.sources[0].origin = Origin.GOOGLE_CTF
    official.writeups = [Writeup(origin=Origin.GOOGLE_CTF, is_inline=False)]
    fork = normalize(_raw(official=False, files=src, spdx=None))
    fork.sources[0].origin = Origin.GITHUB
    fork.writeups = [Writeup(origin=Origin.OTHER, is_inline=False)]

    # same source → same fingerprint → one cluster
    clusters = cluster_by_fingerprint([official, fork])
    assert len(clusters) == 1
    merged = merge_cluster(next(iter(clusters.values())))
    # official won and both provenance edges + writeups were folded in
    assert any(s.origin is Origin.GOOGLE_CTF for s in merged.sources)
    assert any(s.origin is Origin.GITHUB for s in merged.sources)
    assert len(merged.writeups) == 2
    assert merged.license.redistributable  # canonical license is the official MIT one


# --------------------------------------------------------------------------- #
# ratelimit
# --------------------------------------------------------------------------- #


def test_min_interval_is_enforced():
    limiter = RateLimiter(rate=1000, per=1.0, burst=1000, min_interval=0.05)
    limiter.acquire()  # first is free
    start = time.monotonic()
    limiter.acquire()  # must wait ~min_interval
    assert time.monotonic() - start >= 0.045
