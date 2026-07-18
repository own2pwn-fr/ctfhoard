"""Hackropole (ANSSI / France Cybersecurity Challenge) connector.

Hackropole (https://hackropole.fr) is the ANSSI's public archive of past FCSC
(France Cybersecurity Challenge) challenges. It is a **static Hugo site** — there
is no JSON API (``/index.json`` is 404), so this connector enumerates challenges
from the XML sitemap and scrapes each challenge's HTML page. All content is
published under the **Licence Ouverte 2.0 / Etalab 2.0** (SPDX ``etalab-2.0``),
which is redistributable with attribution, so every emitted ``RawChallenge``
carries an official ``etalab-2.0`` :class:`~ctfhoard.schema.Source`.

Site structure (verified live against the current site, 2026-07)::

    https://hackropole.fr/sitemap.xml            # sitemap index -> per-language sitemaps
      -> https://hackropole.fr/fr/sitemap.xml    # French  URLs
      -> https://hackropole.fr/en/sitemap.xml    # English URLs

    Challenge page URL:
      https://hackropole.fr/{fr,en}/challenges/<category>/<slug>/
      e.g. /fr/challenges/crypto/fcsc2026-crypto-a-une-vache-pres/

We crawl a single language (``fr`` by default) so the fr/en mirror of each
challenge is not ingested twice.

Per-challenge HTML signals we rely on (stable ones preferred):

* **Title**       ``<meta property="og:title">`` (falls back to ``<h1>``).
* **Category**    the ``<category>`` path segment of the challenge URL (most
                  stable), with the first ``a.badge`` / ``meta[name=keywords]``
                  as fallbacks.
* **Edition/year** an ``a.badge`` whose href slug carries a 4-digit year
                  (``/fr/fcsc2023`` -> "FCSC 2023", year 2023). Used as the
                  ``event_name`` so challenges group per edition.
* **Tags**        the remaining ``a.badge`` links (their href slug, e.g. ``php``).
* **Difficulty**  the number of filled stars (``<use href="#star-fill">``),
                  reported as ``"<n>/5"``.
* **Description** the text of ``div.markdown`` (the statement column) minus its
                  section headings; empty for image-only statements.
* **Files**       ``ul.list-file li a[href]`` — the official downloadable
                  handouts (sources, binaries, pcaps, and dockerized challenges'
                  ``docker-compose.yml``). The link's ``download`` attribute, when
                  present, is the intended filename (e.g. the URL is
                  ``docker-compose.public.yml`` but it should be saved as
                  ``docker-compose.yml`` so :mod:`ctfhoard.normalize` detects the
                  docker artifact).
* **Author(s)**   the ``.font-monospace`` label next to each ``img.avatar`` under
                  the "Auteur" section (kept in ``extra['authors']`` — the schema
                  has no first-class author field).
* **Writeups**    ``a[href*="/writeups/"]`` links in the "Solutions" table.

Downloaded files land under ``self.workdir/<slug>/`` and ``local_dir`` is set so
``normalize()`` walks them into the file manifest and auto-detects source/docker.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser

from ctfhoard import licenses
from ctfhoard.connectors.base import Connector
from ctfhoard.http import PoliteClient, make_client
from ctfhoard.ratelimit import RateLimiter
from ctfhoard.schema import Origin, RawChallenge, Source, Writeup

#: Root of the static site.
BASE_URL = "https://hackropole.fr"

#: Human-facing event name used when a challenge exposes no FCSC edition badge.
DEFAULT_EVENT_NAME = "Hackropole"

#: Skip (do not download) any single handout larger than this. Static site, small
#: files as a rule, but disk images / pcaps can be large — cap gracefully.
MAX_FILE_BYTES = 50 * 1024 * 1024

#: Matches a 4-digit year embedded in an edition slug/label (fcsc2023, "FCSC 2023").
_YEAR_RE = re.compile(r"(19|20)\d{2}")


class HackropoleConnector(Connector):
    """Emit one ``RawChallenge`` per Hackropole challenge, with its files mirrored."""

    cli_name = "hackropole"
    origin = Origin.HACKROPOLE

    def __init__(self, workdir: Path, limit: int | None = None, lang: str = "fr") -> None:
        """``limit`` caps how many challenges are yielded (incremental/testing);
        ``lang`` selects which per-language sitemap to crawl (default ``fr``, to
        avoid ingesting the fr/en mirror of each challenge twice)."""
        super().__init__(workdir)
        self.limit = limit
        self.lang = lang
        # Gentle on a small static site: ~2 req/s sustained, small burst so a short
        # test run (sitemap + a page + a file) incurs no artificial sleeps. The raw
        # client is reused for streamed file downloads; page GETs go through the
        # polite (retrying, rate-limited) wrapper over the same client + limiter.
        self._http = make_client()
        self._limiter = RateLimiter(rate=2.0, per=1.0, burst=5)
        self._client = PoliteClient(self._http, self._limiter)

    # -- enumeration --------------------------------------------------------

    def _sitemap_url(self) -> str:
        return f"{BASE_URL}/{self.lang}/sitemap.xml"

    def _challenge_urls(self) -> list[str]:
        """Extract every ``/challenges/`` page URL from the language sitemap.

        We parse the ``<loc>`` entries directly (the sitemap is small, flat XML);
        only locs on the challenge path are kept and de-duplicated in order.
        """
        resp = self._client.get(self._sitemap_url())
        resp.raise_for_status()
        locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", resp.text)
        urls = [u for u in locs if "/challenges/" in u]
        # Preserve order, drop dupes.
        return list(dict.fromkeys(urls))

    # -- per-challenge parsing ---------------------------------------------

    @staticmethod
    def _slug_from_url(url: str) -> str:
        return urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]

    @staticmethod
    def _category_from_url(url: str) -> str | None:
        """The ``<category>`` path segment of ``/challenges/<category>/<slug>/``."""
        parts = [p for p in urlparse(url).path.split("/") if p]
        if "challenges" in parts:
            idx = parts.index("challenges")
            if idx + 1 < len(parts) - 1:  # category exists and is not the trailing slug
                return parts[idx + 1]
        return None

    @staticmethod
    def _meta(tree: HTMLParser, key: str) -> str | None:
        node = tree.css_first(f'meta[property="{key}"]') or tree.css_first(f'meta[name="{key}"]')
        if node is None:
            return None
        val = node.attributes.get("content")
        return val.strip() if val else None

    def _parse_badges(self, tree: HTMLParser, category: str | None) -> tuple[str, list[str]]:
        """Split the header ``a.badge`` links into (event_name, tags).

        The badges are: the category, zero or more topic tags, and (usually) the
        FCSC edition. The edition badge is the one whose slug carries a 4-digit
        year; its label becomes the event name. Everything else that is not the
        category becomes a tag (by its href slug, e.g. ``php``).
        """
        event_name = DEFAULT_EVENT_NAME
        tags: list[str] = []
        for a in tree.css("a.badge"):
            href = a.attributes.get("href") or ""
            slug = href.rstrip("/").rsplit("/", 1)[-1]
            label = a.text(strip=True)
            if category and slug == category:
                continue  # the category badge, already captured from the URL
            if _YEAR_RE.search(slug) or _YEAR_RE.search(label):
                event_name = label or event_name
                continue
            if slug:
                tags.append(slug)
        return event_name, tags

    @staticmethod
    def _parse_description(tree: HTMLParser) -> str | None:
        """Statement text: ``div.markdown`` minus its section headings."""
        md = tree.css_first("div.markdown")
        if md is None:
            return None
        for heading in md.css("h2"):
            heading.decompose()
        text = md.text(strip=True)
        return text or None

    @staticmethod
    def _parse_authors(tree: HTMLParser) -> list[str]:
        authors: list[str] = []
        for img in tree.css("img.avatar"):
            parent = img.parent
            name = parent.css_first(".font-monospace") if parent is not None else None
            if name is not None:
                text = name.text(strip=True)
                if text:
                    authors.append(text)
        return authors

    @staticmethod
    def _parse_writeups(tree: HTMLParser, page_url: str) -> list[Writeup]:
        seen: dict[str, None] = {}
        for a in tree.css('a[href*="/writeups/"]'):
            href = a.attributes.get("href")
            if href:
                seen.setdefault(urljoin(page_url, href), None)
        return [Writeup(url=url, origin=Origin.HACKROPOLE, is_inline=False) for url in seen]

    @staticmethod
    def _parse_file_links(tree: HTMLParser, page_url: str) -> list[tuple[str, str]]:
        """Return ``(absolute_url, filename)`` for each official downloadable file.

        The ``download`` attribute (when set) is the intended filename — it can
        differ from the URL basename (``docker-compose.public.yml`` served, saved
        as ``docker-compose.yml``), which matters for docker detection.
        """
        out: list[tuple[str, str]] = []
        for a in tree.css("ul.list-file li a"):
            href = a.attributes.get("href")
            if not href:
                continue
            url = urljoin(page_url, href)
            download = a.attributes.get("download")
            name = (download or a.text(strip=True) or Path(urlparse(url).path).name).strip()
            name = Path(name).name or Path(urlparse(url).path).name  # never a path
            if name:
                out.append((url, name))
        return out

    def _download_files(self, file_links: list[tuple[str, str]], dest: Path) -> None:
        """Download each handout into ``dest`` (skipping ones over the size cap).

        Uses a streamed GET so an over-cap blob is abandoned without buffering it
        whole. Failures on individual files are swallowed — a missing handout must
        not sink the whole challenge record.
        """
        for url, name in file_links:
            target = dest / name
            if target.exists() and target.stat().st_size > 0:
                continue  # resume: reuse cached download
            try:
                self._stream_to_file(url, target)
            except Exception:  # noqa: BLE001 — one bad file must not drop the challenge
                target.unlink(missing_ok=True)

    def _stream_to_file(self, url: str, target: Path) -> None:
        """Stream ``url`` into ``target``, honoring the rate limiter and size cap.

        A file whose declared ``Content-Length`` exceeds the cap is skipped before
        any body is read; a file that grows past the cap mid-stream is abandoned.
        """
        self._limiter.acquire()
        with self._http.stream("GET", url) as resp:
            resp.raise_for_status()
            declared = resp.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > MAX_FILE_BYTES:
                return  # too large — do not materialize
            written = 0
            with target.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    written += len(chunk)
                    if written > MAX_FILE_BYTES:
                        fh.close()
                        target.unlink(missing_ok=True)
                        return
                    fh.write(chunk)

    def _parse_challenge(self, url: str, html: str) -> RawChallenge:
        tree = HTMLParser(html)
        slug = self._slug_from_url(url)
        category = self._category_from_url(url) or (self._meta(tree, "keywords") or "").split(",")[
            0
        ].strip() or None

        title = self._meta(tree, "og:title")
        if not title:
            h1 = tree.css_first("h1")
            title = h1.text(strip=True) if h1 else slug

        event_name, tags = self._parse_badges(tree, category)
        year_match = _YEAR_RE.search(event_name) or _YEAR_RE.search(slug)
        year = int(year_match.group(0)) if year_match else None

        stars = len(tree.css('use[href="#star-fill"]'))
        difficulty = f"{stars}/5" if stars else None

        dest = self.workdir / slug
        dest.mkdir(parents=True, exist_ok=True)
        self._download_files(self._parse_file_links(tree, url), dest)

        return RawChallenge(
            origin=Origin.HACKROPOLE,
            title=title,
            event_name=event_name,
            year=year,
            raw_category=category,
            tags=tags,
            difficulty=difficulty,
            description=self._parse_description(tree),
            local_dir=str(dest),
            writeups=self._parse_writeups(tree, url),
            source=Source(
                origin=Origin.HACKROPOLE,
                url=url,
                is_official=True,
                license=licenses.from_spdx(
                    "etalab-2.0", note="Hackropole content under Licence Ouverte 2.0"
                ),
            ),
            extra={"slug": slug, "authors": self._parse_authors(tree)},
        )

    # -- driver -------------------------------------------------------------

    def discover(self) -> Iterator[RawChallenge]:
        urls = self._challenge_urls()
        if self.limit is not None:
            urls = urls[: self.limit]
        for url in urls:
            resp = self._client.get(url)
            if resp.status_code != 200:
                continue  # deleted/redirected page — skip, don't abort the crawl
            yield self._parse_challenge(url, resp.text)


CONNECTOR = HackropoleConnector
