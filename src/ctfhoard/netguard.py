"""SSRF-safe egress guard for fetching untrusted, connector-supplied URLs.

External writeup links come from arbitrary third parties (CTFtime entries, random
blogs, GitHub READMEs). Fetching them blindly is a Server-Side Request Forgery
vector: a hostile writeup URL pointing at ``http://169.254.169.254/`` (cloud
metadata) or ``http://127.0.0.1:6379/`` (a local Redis), or a public URL that
302-redirects there, would have its response body written into ``data/corpus/``
and later published to a PUBLIC dataset — leaking cloud credentials or internal
data. This module gates every such fetch:

* :func:`assert_safe_url` / :func:`is_public_http_url` reject non-http(s) schemes
  and resolve the hostname, refusing the request if ANY resolved IP is
  loopback / link-local / private / reserved / multicast / unspecified.
* :func:`safe_get` validates the URL, disables automatic redirects, manually
  follows a bounded number of redirects RE-VALIDATING each hop's ``Location``,
  and streams the body under a hard byte cap so a multi-GB or gzip-bomb response
  cannot exhaust memory.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

# Hard cap on a single fetched writeup body (mirrors hackropole's MAX_FILE_BYTES
# intent): reject before reading when Content-Length declares more, and abort
# mid-stream once accumulated bytes exceed it.
MAX_WRITEUP_BYTES = 25 * 1024 * 1024  # 25 MiB
# Maximum number of redirect hops to follow (each re-validated).
MAX_REDIRECTS = 5
DEFAULT_TIMEOUT = 30.0

_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


class UnsafeUrlError(ValueError):
    """Raised when a URL is not a public http(s) URL safe to fetch."""


def _addr_is_public(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True only for globally-routable unicast addresses.

    IPv4-mapped IPv6 addresses (e.g. ``::ffff:169.254.169.254``) are unwrapped so
    they cannot be used to smuggle an internal IPv4 target past the checks.
    """
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return not (
        addr.is_loopback
        or addr.is_link_local
        or addr.is_private
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def assert_safe_url(url: str) -> None:
    """Raise :class:`UnsafeUrlError` unless ``url`` is a public http(s) URL.

    Rejects non-http(s) schemes, missing hosts, unresolvable hosts, and any host
    that resolves (via ``socket.getaddrinfo``) to a non-public IP address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"non-http(s) scheme: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeUrlError(f"missing host in url: {url!r}")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"cannot resolve host {host!r}: {exc}") from exc
    if not infos:
        raise UnsafeUrlError(f"no addresses for host {host!r}")

    for info in infos:
        sockaddr = info[4]
        ip = ipaddress.ip_address(sockaddr[0])
        if not _addr_is_public(ip):
            raise UnsafeUrlError(f"host {host!r} resolves to non-public address {ip}")


def is_public_http_url(url: str) -> bool:
    """Boolean form of :func:`assert_safe_url` — True iff ``url`` is safe to fetch."""
    try:
        assert_safe_url(url)
    except UnsafeUrlError:
        return False
    return True


def safe_get(
    client,
    url: str,
    *,
    max_bytes: int = MAX_WRITEUP_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, bytes, str | None]:
    """Fetch ``url`` through ``client``, SSRF-guarded and byte-capped.

    ``client`` must expose httpx.Client's streaming API:
    ``client.stream("GET", url, follow_redirects=False, timeout=...)`` returning a
    context-managed response with ``.status_code``, ``.headers`` and
    ``.iter_bytes()`` (and a ``Location`` header on redirects).

    Redirects are followed manually (auto-redirects disabled) up to
    :data:`MAX_REDIRECTS`, re-validating each hop's resolved target before it is
    fetched. Returns ``(status_code, body, content_type)``.

    Raises :class:`UnsafeUrlError` when the URL — or any redirect hop — is not a
    public http(s) URL, when too many redirects are chained, or when the declared
    or streamed body exceeds ``max_bytes``.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        assert_safe_url(current)
        with client.stream("GET", current, follow_redirects=False, timeout=timeout) as resp:
            if resp.status_code in _REDIRECT_STATUS:
                location = resp.headers.get("location")
                if not location:
                    raise UnsafeUrlError(f"redirect without Location from {current!r}")
                # Resolve relative redirects against the current URL, then re-check.
                current = urljoin(current, location)
                continue

            declared = resp.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                raise UnsafeUrlError(
                    f"declared Content-Length {declared} exceeds cap {max_bytes}"
                )

            body = bytearray()
            for chunk in resp.iter_bytes():
                body += chunk
                if len(body) > max_bytes:
                    raise UnsafeUrlError(f"response body exceeds cap {max_bytes}")
            return resp.status_code, bytes(body), resp.headers.get("content-type")

    raise UnsafeUrlError(f"too many redirects (> {MAX_REDIRECTS}) starting at {url!r}")
