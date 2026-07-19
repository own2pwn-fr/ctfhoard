"""Tests for the corpus materializer: hard copies of sources + writeups on disk.

Fully offline: every fetch goes through ``ctfhoard.netguard``, whose DNS
resolution (``socket.getaddrinfo``) is monkeypatched so no real network or name
service is touched.
"""

from __future__ import annotations

import socket
import tarfile

import pytest

from ctfhoard import netguard
from ctfhoard.corpus import (
    archive_challenge,
    challenge_relpath,
    materialize_and_archive,
    materialize_challenge,
)
from ctfhoard.netguard import UnsafeUrlError, is_public_http_url, safe_get
from ctfhoard.schema import Challenge, FileEntry, Origin, Source, Writeup

# Hosts the fake resolver treats as public (globally-routable). Anything else is
# resolved as a literal IP if possible, otherwise reported unresolvable.
_PUBLIC_HOSTS = {"blog.example": "93.184.216.34", "public.example": "93.184.216.34"}


def _fake_getaddrinfo(host, port, *args, **kwargs):
    """Offline stand-in for ``socket.getaddrinfo``.

    Known test hosts resolve to a public IP; literal IPs resolve to themselves;
    everything else raises ``gaierror`` (as a real unresolvable host would).
    """
    ip = _PUBLIC_HOSTS.get(host)
    if ip is None:
        try:
            import ipaddress

            ipaddress.ip_address(host)  # literal IP address?
            ip = host
        except ValueError as exc:
            raise socket.gaierror(f"unknown host {host!r}") from exc
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch):
    monkeypatch.setattr(netguard.socket, "getaddrinfo", _fake_getaddrinfo)


class _FakeStream:
    """Context-managed stand-in for an ``httpx`` streaming response."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.status_code = status
        self._body = body
        self.headers = headers

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def iter_bytes(self, chunk_size: int = 65536):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class _FakeClient:
    """Stand-in for the underlying httpx client — records fetched URLs.

    ``routes`` maps a URL to ``(status, body, headers)``; unmapped URLs return a
    default 200 HTML body.
    """

    def __init__(self, routes: dict[str, tuple[int, bytes, dict[str, str]]] | None = None) -> None:
        self.fetched: list[str] = []
        self.routes = routes or {}

    def stream(self, method: str, url: str, *, follow_redirects: bool = False, timeout=None):
        self.fetched.append(url)
        if url in self.routes:
            status, body, headers = self.routes[url]
            return _FakeStream(status, body, headers)
        return _FakeStream(
            200,
            b"<html><body>external writeup body</body></html>",
            {"content-type": "text/html"},
        )


def _challenge_with_sources(raw_dir) -> Challenge:
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


def test_materialize_without_client_skips_external_but_keeps_inline(tmp_path):
    raw = tmp_path / "raw"
    ch = _challenge_with_sources(raw)
    materialize_challenge(ch, tmp_path / "corpus", raw_dir=raw, client=None)

    dest = tmp_path / "corpus" / challenge_relpath(ch)
    assert (dest / "writeups" / "writeup_00.md").exists()  # inline still written
    assert not (dest / "writeups" / "writeup_01.html").exists()  # external skipped
    assert ch.writeups[1].local_path is None


# --------------------------------------------------------------------------
# Finding 1 — SSRF: external writeup URLs (and redirects) resolving to
# internal/link-local/loopback IPs must be refused, never written to the corpus.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "http://127.0.0.1:6379/",  # local Redis (loopback)
        "http://10.0.0.5/internal",  # private
        "http://[::1]/",  # IPv6 loopback
        "file:///etc/passwd",  # non-http scheme
        "gopher://127.0.0.1:11211/",  # non-http scheme
    ],
)
def test_ssrf_unsafe_urls_are_rejected(url):
    assert is_public_http_url(url) is False
    with pytest.raises(UnsafeUrlError):
        netguard.assert_safe_url(url)


def test_ssrf_public_url_is_accepted():
    assert is_public_http_url("https://blog.example/writeup") is True


def test_materialize_skips_ssrf_writeup_without_crashing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    ch = Challenge(
        id="ssrf00000000000000000000",
        title="SSRF Bait",
        event_name="Test CTF",
        year=2024,
        sources=[Source(origin=Origin.GITHUB, repo="x/y")],
        writeups=[Writeup(origin=Origin.GITHUB, url="http://169.254.169.254/latest/meta-data/")],
    )
    client = _FakeClient()
    materialize_challenge(ch, tmp_path / "corpus", raw_dir=raw, client=client, repo_root=tmp_path)

    dest = tmp_path / "corpus" / challenge_relpath(ch)
    # No body was fetched or written — the internal target was refused up front.
    assert client.fetched == []
    assert not (dest / "writeups" / "writeup_00.html").exists()
    assert ch.writeups[0].local_path is None


def test_safe_get_revalidates_redirect_to_internal():
    # A public URL that 302-redirects to a link-local metadata endpoint: the hop
    # must be re-validated and refused (the classic SSRF-via-redirect bypass).
    routes = {
        "http://public.example/wu": (
            302,
            b"",
            {"location": "http://169.254.169.254/latest/meta-data/"},
        )
    }
    client = _FakeClient(routes)
    with pytest.raises(UnsafeUrlError):
        safe_get(client, "http://public.example/wu")
    # The internal redirect target was never actually fetched.
    assert client.fetched == ["http://public.example/wu"]


# --------------------------------------------------------------------------
# Finding 2 — writeup download must be streamed under a hard byte cap so a
# multi-GB / gzip-bomb response cannot exhaust memory.
# --------------------------------------------------------------------------


def test_safe_get_aborts_when_body_exceeds_cap():
    big = b"A" * 4096
    routes = {"http://public.example/big": (200, big, {"content-type": "text/plain"})}
    client = _FakeClient(routes)
    with pytest.raises(UnsafeUrlError):
        safe_get(client, "http://public.example/big", max_bytes=1024)


def test_safe_get_rejects_declared_content_length_over_cap():
    routes = {
        "http://public.example/huge": (
            200,
            b"tiny actual body",
            {"content-type": "text/plain", "content-length": str(100 * 1024 * 1024)},
        )
    }
    client = _FakeClient(routes)
    with pytest.raises(UnsafeUrlError):
        safe_get(client, "http://public.example/huge", max_bytes=1024)


def test_safe_get_returns_body_under_cap():
    routes = {"http://public.example/ok": (200, b"hello", {"content-type": "text/plain"})}
    client = _FakeClient(routes)
    status, body, ctype = safe_get(client, "http://public.example/ok", max_bytes=1024)
    assert (status, body, ctype) == (200, b"hello", "text/plain")


# --------------------------------------------------------------------------
# Finding 3 — corpus_path must be stored RELATIVE to data_dir even when the
# data dir is an absolute path (no machine path leaked into the catalog).
# --------------------------------------------------------------------------


def test_corpus_path_is_relative_to_absolute_data_dir(tmp_path):
    data_dir = (tmp_path / "data").resolve()
    corpus_root = data_dir / "corpus"
    raw = tmp_path / "raw"
    ch = _challenge_with_sources(raw)

    materialize_challenge(ch, corpus_root, raw_dir=raw, client=None, repo_root=data_dir)

    assert ch.corpus_path is not None
    # Relative, POSIX-ish, under corpus/, and no absolute machine path.
    rel_posix = ch.corpus_path.replace("\\", "/")
    assert rel_posix.startswith("corpus/")
    assert not rel_posix.startswith("/")
    assert str(data_dir) not in ch.corpus_path


# --------------------------------------------------------------------------
# Finding 4 — re-ingestion must not leave orphan writeup files that no catalog
# entry references (corpus <-> catalog divergence).
# --------------------------------------------------------------------------


def test_reingest_prunes_orphan_writeups(tmp_path):
    raw = tmp_path / "raw"
    corpus = tmp_path / "corpus"

    # First ingest: two writeups -> writeup_00.md + writeup_01.html.
    ch1 = _challenge_with_sources(raw)
    materialize_challenge(ch1, corpus, raw_dir=raw, client=_FakeClient(), repo_root=tmp_path)
    dest = corpus / challenge_relpath(ch1)
    assert (dest / "writeups" / "writeup_01.html").exists()

    # Re-ingest the SAME challenge (same id/relpath) with only the inline writeup.
    ch2 = _challenge_with_sources(raw)
    ch2.writeups = [Writeup(origin=Origin.CTFTIME, is_inline=True, text="# Solution\nonly inline")]
    materialize_challenge(ch2, corpus, raw_dir=raw, client=_FakeClient(), repo_root=tmp_path)

    # The stale external writeup file must be gone; corpus matches the catalog.
    assert (dest / "writeups" / "writeup_00.md").exists()
    assert not (dest / "writeups" / "writeup_01.html").exists()
    assert {f.path for f in ch2.files if f.path.startswith("writeups/")} == {
        "writeups/writeup_00.md"
    }


# --------------------------------------------------------------------------
# Finding 5 — path safety: hostile titles must not escape the corpus root, and
# symlinked sources must never be copied out of raw_dir.
# --------------------------------------------------------------------------


def test_relpath_hostile_title_stays_under_corpus_root(tmp_path):
    corpus_root = (tmp_path / "corpus").resolve()
    ch = Challenge(
        id="ffffffffffffffffffffffff",
        title="../../../etc/evil",
        event_name="..",
        year=2024,
        sources=[Source(origin=Origin.GITHUB, repo="x/y")],
    )
    rel = challenge_relpath(ch)
    assert ".." not in rel.parts
    resolved = (corpus_root / rel).resolve()
    assert resolved.is_relative_to(corpus_root)


def test_materialize_skips_symlinked_sources_escaping_raw_dir(tmp_path):
    # Secret content that lives OUTSIDE the challenge's raw dir.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOPSECRET")

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "chal.py").write_text("print('ok')\n")
    # A symlinked file pointing at a host file (e.g. /etc/passwd).
    (raw / "passwd_link").symlink_to("/etc/passwd")
    # A symlinked file pointing outside raw_dir.
    (raw / "leak.txt").symlink_to(outside / "secret.txt")
    # A symlinked DIRECTORY escape: raw/evil -> outside/.
    (raw / "evil").symlink_to(outside, target_is_directory=True)

    ch = Challenge(
        id="aaaabbbbccccddddeeeeffff",
        title="Symlink Escape",
        event_name="Test CTF",
        year=2024,
        sources=[Source(origin=Origin.GITHUB, repo="x/y")],
    )
    corpus = tmp_path / "corpus"
    materialize_challenge(ch, corpus, raw_dir=raw, client=None, repo_root=tmp_path)

    dest = corpus / challenge_relpath(ch)
    # The legit source copied; every symlinked escape refused.
    assert (dest / "chal.py").exists()
    assert not (dest / "passwd_link").exists()
    assert not (dest / "leak.txt").exists()
    assert not (dest / "evil").exists()
    assert not (dest / "evil" / "secret.txt").exists()
    # Nothing containing the secret was written anywhere under the corpus root.
    for p in corpus.rglob("*"):
        if p.is_file():
            assert b"TOPSECRET" not in p.read_bytes()


def test_materialize_survives_symlinked_dir_in_sources(tmp_path):
    # Regression: real repos (e.g. google-ctf) ship symlinked directories such as
    # `exploit -> ../solution`. rglob would follow them, duplicating files and
    # triggering a FileExistsError on mkdir when a name is a file here and a dir
    # there. _copy_sources must walk with followlinks=False and never crash.
    raw = tmp_path / "raw"
    (raw / "solution").mkdir(parents=True)
    (raw / "solution" / "sploit.py").write_text("# solve\n")
    (raw / "chal.c").write_text("int main(){}\n")
    # a symlinked directory pointing elsewhere in the tree
    (raw / "exploit").symlink_to(raw / "solution", target_is_directory=True)

    ch = Challenge(
        id="ffeeddccbbaa998877665544",
        title="Sym Chall",
        event_name="Test CTF",
        year=2024,
        sources=[Source(origin=Origin.GITHUB, repo="x/y")],
    )
    # must not raise
    materialize_challenge(ch, tmp_path / "corpus", raw_dir=raw, client=None)

    dest = tmp_path / "corpus" / challenge_relpath(ch)
    assert (dest / "chal.c").exists()
    assert (dest / "solution" / "sploit.py").exists()
    # the symlinked dir itself is not copied/traversed as a real directory
    assert not (dest / "exploit").is_dir()


# --------------------------------------------------------------------------
# Per-challenge archiving: pack each materialized challenge dir into ONE
# deterministic .tar.gz (HF per-file upload rate limit workaround).
# --------------------------------------------------------------------------


def _build_challenge_dir(root):
    """A small fake materialized challenge tree (nested dirs + a writeup)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "chal.py").write_text("print('pwn')\n")
    (root / "Dockerfile").write_text("FROM scratch\n")
    (root / "src").mkdir()
    (root / "src" / "helper.c").write_text("int main(){}\n")
    (root / "writeups").mkdir()
    (root / "writeups" / "writeup_00.md").write_text("# Solution\nbody\n")


def test_archive_challenge_packs_and_removes_loose_dir(tmp_path):
    cdir = tmp_path / "chal__deadbeef"
    _build_challenge_dir(cdir)

    archive = archive_challenge(cdir)

    # A single .tar.gz remains; the loose dir is gone.
    assert archive == tmp_path / "chal__deadbeef.tar.gz"
    assert archive.exists()
    assert not cdir.exists()

    # Extracting reproduces the exact tree at paths relative to the challenge dir.
    extract = tmp_path / "extract"
    with tarfile.open(archive, "r:gz") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile())
        tar.extractall(extract, filter="data")
    assert names == [
        "Dockerfile",
        "chal.py",
        "src/helper.c",
        "writeups/writeup_00.md",
    ]
    assert (extract / "chal.py").read_text() == "print('pwn')\n"
    assert (extract / "src" / "helper.c").read_text() == "int main(){}\n"
    assert (extract / "writeups" / "writeup_00.md").read_text().startswith("# Solution")


def test_archive_challenge_is_byte_deterministic(tmp_path):
    # Two identical challenge trees must archive to byte-identical tarballs so HF
    # diffing skips unchanged challenges on re-runs.
    a = tmp_path / "a" / "chal__cafe"
    b = tmp_path / "b" / "chal__cafe"
    _build_challenge_dir(a)
    _build_challenge_dir(b)

    arc_a = archive_challenge(a)
    arc_b = archive_challenge(b)

    assert arc_a.read_bytes() == arc_b.read_bytes()


def test_materialize_and_archive_repoints_corpus_path(tmp_path):
    raw = tmp_path / "raw"
    ch = _challenge_with_sources(raw)
    corpus = tmp_path / "corpus"

    materialize_and_archive(ch, corpus, raw_dir=raw, client=_FakeClient(), repo_root=tmp_path)

    # The loose challenge dir was replaced by one tarball; corpus_path points at it.
    dest = corpus / challenge_relpath(ch)
    archive = dest.with_name(dest.name + ".tar.gz")
    assert not dest.exists()
    assert archive.exists()
    assert ch.corpus_path is not None
    assert ch.corpus_path.endswith(".tar.gz")
    assert ch.corpus_path == str(archive.relative_to(tmp_path))

    # The per-file browsable manifest (sources + writeups) is preserved in the catalog.
    paths = {f.path for f in ch.files}
    assert {"chal.py", "Dockerfile", "writeups/writeup_00.md"} <= paths

    # The archive is independently extractable and contains the writeup bytes.
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name for m in tar.getmembers() if m.isfile()}
    assert {"chal.py", "Dockerfile", "writeups/writeup_00.md"} <= members
