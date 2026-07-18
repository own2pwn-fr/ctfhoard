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
from ctfhoard.normalize import (
    build_file_manifest,
    map_category,
    normalize,
    slugify,
    synthesize_id,
)
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


def test_slugify_is_unicode_aware_but_ascii_stable():
    # ASCII behavior is unchanged.
    assert slugify("HITCON CTF") == "hitcon-ctf"
    assert slugify("Baby Pwn") == "baby-pwn"
    # Non-ASCII word characters survive instead of collapsing to '' (regression:
    # the old ``[^a-z0-9]+`` rule wiped CJK/Cyrillic titles entirely).
    assert slugify("你好世界")
    assert slugify("Привет мир")


def test_synthesize_id_distinct_for_non_ascii_titles():
    # Two DIFFERENT CJK titles in the same event/year/category must not collide.
    # Under the old slugify both collapsed to '' -> identical basis -> identical id,
    # so merge_jsonl (last-write-wins by id) would silently drop one challenge.
    a = synthesize_id("Some CTF", 2024, "你好世界", "web")
    b = synthesize_id("Some CTF", 2024, "再见世界", "web")
    assert a != b

    # Emoji/punctuation-only titles have an empty slug but must still be distinct,
    # anchored on the raw-title hash.
    e1 = synthesize_id("Some CTF", 2024, "🎉🎉", "web")
    e2 = synthesize_id("Some CTF", 2024, "🚩🚩", "web")
    assert e1 != e2


def test_normalize_slug_non_empty_for_non_ascii_title():
    ch = normalize(_raw(title="你好世界"))
    assert ch.slug  # non-empty slug for a CJK title
    # Emoji-only title still yields a usable (non-empty) slug via the id fallback.
    assert normalize(_raw(title="🎉")).slug


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


def test_manifest_hashes_large_source_files_and_keeps_lfs(tmp_path):
    # Regression: build_file_manifest used to leave sha256='' for files over the size
    # cap. Two challenges sharing scaffolding but differing only in a large source
    # file then produced the SAME content fingerprint (the big file was dropped) and
    # were wrongly merged. Now large files are hashed AND still flagged lfs.
    def build(dir_name: str, big_content: bytes):
        d = tmp_path / dir_name
        d.mkdir()
        (d / "Dockerfile").write_text("FROM scratch\n")
        (d / "disk.img").write_bytes(big_content)
        return build_file_manifest(d, size_cap=8)  # tiny cap -> disk.img is "big"

    files_a = build("a", b"X" * 4096)
    files_b = build("b", b"Y" * 4096)

    big_a = next(f for f in files_a if f.path == "disk.img")
    big_b = next(f for f in files_b if f.path == "disk.img")
    assert big_a.lfs and big_b.lfs  # still marked for Git LFS
    assert big_a.sha256 and big_b.sha256  # but hashed, not dropped
    assert big_a.sha256 != big_b.sha256

    fp_a = content_fingerprint(files_a)
    fp_b = content_fingerprint(files_b)
    assert fp_a and fp_b
    assert fp_a != fp_b  # distinct large source -> distinct fingerprint (no false merge)


def test_fingerprint_anchors_source_files_without_sha():
    # Defensive fallback in content_fingerprint: even if a source file carries no
    # sha256 (e.g. an LFS pointer that was never hashed), it still anchors on lfs_oid
    # so two challenges differing only in that file don't collapse together.
    base = FileEntry(path="Dockerfile", sha256="d" * 64, size=10, is_source=True)
    big_a = FileEntry(
        path="disk.img", sha256="", size=60 * 1024 * 1024, is_source=True,
        lfs=True, lfs_oid="oid-aaaa",
    )
    big_b = FileEntry(
        path="disk.img", sha256="", size=60 * 1024 * 1024, is_source=True,
        lfs=True, lfs_oid="oid-bbbb",
    )
    fp_a = content_fingerprint([base, big_a])
    fp_b = content_fingerprint([base, big_b])
    assert fp_a and fp_b
    assert fp_a != fp_b


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


def test_merge_three_mirrors_dedups_edges_by_identity():
    src = [FileEntry(path="chal.py", sha256="a" * 64, size=1, is_source=True)]

    def mirror() -> object:
        return normalize(_raw(official=False, files=src, spdx=None))

    # A source edge shared verbatim by two mirrors, and a writeup shared by them too.
    def shared_source() -> Source:
        return Source(
            origin=Origin.GITHUB,
            url="https://github.com/o/r",
            repo="o/r",
            path_in_repo="chal",
        )

    shared_wu_url = "https://blog.example/writeup"

    m1 = mirror()
    m1.sources = [shared_source()]
    m1.writeups = [Writeup(origin=Origin.GITHUB, url=shared_wu_url)]

    m2 = mirror()
    m2.sources = [shared_source()]  # byte-identical provenance edge
    m2.writeups = [Writeup(origin=Origin.GITHUB, url=shared_wu_url)]  # identical writeup

    m3 = mirror()
    m3.sources = [
        Source(origin=Origin.SAJJADIUM, url="https://arch.example/x", repo="a/b", path_in_repo="c"),
        # Same repo/url as the shared edge but a DIFFERENT path -> distinct, must NOT collapse.
        Source(
            origin=Origin.GITHUB, url="https://github.com/o/r", repo="o/r", path_in_repo="chal2"
        ),
    ]
    m3.writeups = [Writeup(origin=Origin.OTHER, url="https://blog.example/other")]

    clusters = cluster_by_fingerprint([m1, m2, m3])
    assert len(clusters) == 1  # identical source set -> one cluster
    merged = merge_cluster(next(iter(clusters.values())))

    edges = [(s.origin, s.repo, s.path_in_repo) for s in merged.sources]
    assert edges.count((Origin.GITHUB, "o/r", "chal")) == 1  # identical edge folded once
    assert (Origin.GITHUB, "o/r", "chal2") in edges  # same url/repo, different path -> kept
    assert (Origin.SAJJADIUM, "a/b", "c") in edges  # distinct edge kept
    assert len(merged.sources) == 3

    wu_keys = [(w.origin, str(w.url)) for w in merged.writeups]
    assert wu_keys.count((Origin.GITHUB, str(m1.writeups[0].url))) == 1  # deduped by (origin,url)
    assert len(merged.writeups) == 2  # shared + distinct


# --------------------------------------------------------------------------- #
# ratelimit
# --------------------------------------------------------------------------- #


def test_min_interval_is_enforced():
    limiter = RateLimiter(rate=1000, per=1.0, burst=1000, min_interval=0.05)
    limiter.acquire()  # first is free
    start = time.monotonic()
    limiter.acquire()  # must wait ~min_interval
    assert time.monotonic() - start >= 0.045


def test_token_bucket_refills_and_caps_at_capacity():
    # 2 tokens/sec, burst 2, no min-interval floor. Drain the burst, then the 3rd
    # acquire must wait for one token to refill: deficit(1) * per/rate = 0.5s.
    limiter = RateLimiter(rate=2, per=1.0, burst=2, min_interval=0)
    assert limiter.acquire() == 0.0  # burst token 1 (free)
    assert limiter.acquire() == 0.0  # burst token 2 (free)

    start = time.monotonic()
    slept = limiter.acquire()  # tokens exhausted -> must sleep ~0.5s
    elapsed = time.monotonic() - start
    assert 0.4 <= elapsed <= 0.9
    assert slept >= 0.4

    # Refill is capped at capacity: a long idle period never accrues a burst
    # beyond ``burst`` tokens.
    limiter._last_refill = time.monotonic() - 100.0
    limiter._refill(time.monotonic())
    assert limiter._tokens == limiter.capacity
