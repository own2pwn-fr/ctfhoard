"""Tests for the corpus materializer: hard copies of sources + writeups on disk."""

from __future__ import annotations

from pathlib import Path

from ctfhoard.corpus import challenge_relpath, materialize_challenge
from ctfhoard.schema import Challenge, FileEntry, Origin, Source, Writeup


class _FakeResp:
    def __init__(self, content: bytes, content_type: str = "text/html") -> None:
        self.status_code = 200
        self.content = content
        self.headers = {"content-type": content_type}


class _FakeClient:
    """Stand-in for PoliteClient — records fetched URLs, returns canned bytes."""

    def __init__(self) -> None:
        self.fetched: list[str] = []

    def get(self, url: str):
        self.fetched.append(url)
        return _FakeResp(b"<html><body>external writeup body</body></html>")


def _challenge_with_sources(raw_dir: Path) -> Challenge:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "chal.py").write_text("print('pwn')\n")
    (raw_dir / "Dockerfile").write_text("FROM scratch\n")
    return Challenge(
        id="abcdef1234567890abcdef12",
        title="Baby Pwn",
        event_name="Test CTF",
        year=2024,
        sources=[Source(origin=Origin.GITHUB, repo="x/y")],
        files=[
            FileEntry(path="chal.py", sha256="a" * 64, size=12, is_source=True),
            FileEntry(path="Dockerfile", sha256="b" * 64, size=11, is_source=True),
        ],
        writeups=[
            Writeup(origin=Origin.CTFTIME, is_inline=True, text="# Solution\nInline body."),
            Writeup(origin=Origin.GITHUB, is_inline=False, url="https://blog.example/wu"),
        ],
    )


def test_relpath_is_deterministic(tmp_path):
    ch = _challenge_with_sources(tmp_path / "raw")
    p1 = challenge_relpath(ch)
    p2 = challenge_relpath(ch)
    assert p1 == p2
    assert p1.as_posix() == "github/test-ctf/2024/baby-pwn__abcdef12"


def test_materialize_copies_sources_and_writeups(tmp_path):
    raw = tmp_path / "raw"
    ch = _challenge_with_sources(raw)
    corpus = tmp_path / "corpus"
    client = _FakeClient()

    materialize_challenge(ch, corpus, raw_dir=raw, client=client, repo_root=tmp_path)

    dest = corpus / challenge_relpath(ch)
    # sources copied in verbatim
    assert (dest / "chal.py").read_text() == "print('pwn')\n"
    assert (dest / "Dockerfile").exists()

    # inline writeup written as .md, external writeup fetched + written as .html
    inline = dest / "writeups" / "writeup_00.md"
    external = dest / "writeups" / "writeup_01.html"
    assert inline.read_text().startswith("# Solution")
    assert b"external writeup body" in external.read_bytes()
    assert client.fetched == ["https://blog.example/wu"]

    # catalog stays lean: text dropped, local_path set, corpus_path relative to repo
    assert all(w.text is None for w in ch.writeups)
    assert ch.writeups[0].local_path == "writeups/writeup_00.md"
    assert ch.writeups[1].local_path == "writeups/writeup_01.html"
    assert ch.corpus_path == str((corpus / challenge_relpath(ch)).relative_to(tmp_path))

    # writeup files were added to the manifest
    manifest_paths = {f.path for f in ch.files}
    assert "writeups/writeup_00.md" in manifest_paths
    assert "writeups/writeup_01.html" in manifest_paths


def test_materialize_without_client_skips_external_but_keeps_inline(tmp_path):
    raw = tmp_path / "raw"
    ch = _challenge_with_sources(raw)
    materialize_challenge(ch, tmp_path / "corpus", raw_dir=raw, client=None)

    dest = tmp_path / "corpus" / challenge_relpath(ch)
    assert (dest / "writeups" / "writeup_00.md").exists()  # inline still written
    assert not (dest / "writeups" / "writeup_01.html").exists()  # external skipped
    assert ch.writeups[1].local_path is None
