"""OWASP Juice Shop connector.

Juice Shop ships a single, clean, machine-readable spec of every built-in
challenge at ``data/static/challenges.yml`` in its repo. That one file is the
whole source of truth for this connector: no cloning, no scraping, one HTTP GET
and a YAML parse. The app itself is MIT-licensed and fully redistributable, so
every ``RawChallenge`` we emit carries an MIT :class:`~ctfhoard.schema.Source`.

Shape of a ``challenges.yml`` entry (verified against master, 113 challenges)::

    - name: 'Admin Section'
      category: 'Broken Access Control'
      tags:                 # optional; a YAML list (~2/3 of entries have it)
        - Good for Demos
      description: 'Access the administration section of the store.'
      difficulty: 2         # int 1..6
      hints:                # optional; a YAML list of free-text tips
        - 'It is just slightly harder to find than the score board link.'
      mitigationUrl: 'https://.../Authorization_Cheat_Sheet.html'  # optional
      key: adminSectionChallenge

We map ``mitigationUrl`` to a (reference-only) :class:`Writeup`: it is the
official pointer to how the class of bug is fixed. ``tags`` is normalized to a
list, tolerating the comma-separated-string form older revisions used. Fields the
task brief guessed at but the current file does *not* have (``hint`` singular,
``hintUrl``, OWASP/CWE mappings) are handled defensively should they reappear.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from ctfhoard import licenses
from ctfhoard.connectors.base import Connector
from ctfhoard.http import make_client
from ctfhoard.schema import Origin, RawChallenge, Source, Writeup

#: Canonical location of the challenge spec (raw.githubusercontent, master branch).
CHALLENGES_URL = (
    "https://raw.githubusercontent.com/juice-shop/juice-shop/master/data/static/challenges.yml"
)

#: Human-facing event name every Juice Shop challenge is grouped under.
EVENT_NAME = "OWASP Juice Shop"


def _normalize_tags(raw_tags: Any) -> list[str]:
    """Coerce the ``tags`` field into a clean list of strings.

    Current revisions use a YAML list, but the field is optional and older
    revisions used a comma-separated string — accept both and drop blanks.
    """
    if raw_tags is None:
        return []
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",") if t.strip()]
    if isinstance(raw_tags, (list, tuple)):
        return [str(t).strip() for t in raw_tags if str(t).strip()]
    return [str(raw_tags).strip()]


def _build_writeups(entry: dict[str, Any]) -> list[Writeup]:
    """Reference-only writeups from the entry's mitigation/hint URLs.

    ``mitigationUrl`` is the official "how this bug is fixed" pointer. ``hintUrl``
    is not present on current master but was on older revisions, so we still pick
    it up if it reappears. Both are links, not inline text (``is_inline=False``).
    """
    writeups: list[Writeup] = []
    for field, title in (("mitigationUrl", "Mitigation"), ("hintUrl", "Hint")):
        url = entry.get(field)
        if url:
            writeups.append(
                Writeup(
                    url=str(url),
                    origin=Origin.JUICESHOP,
                    title=title,
                    is_inline=False,
                )
            )
    return writeups


class JuiceShopConnector(Connector):
    """Emit one ``RawChallenge`` per built-in OWASP Juice Shop challenge."""

    cli_name = "juiceshop"
    origin = Origin.JUICESHOP

    #: Overridable so tests (or a pinned mirror) can point at a local/alternate URL.
    challenges_url = CHALLENGES_URL

    def __init__(self, workdir: Path, *, challenges_url: str | None = None) -> None:
        super().__init__(workdir)
        if challenges_url is not None:
            self.challenges_url = challenges_url

    def _fetch(self) -> str:
        """Download the raw ``challenges.yml`` text."""
        with make_client() as client:
            resp = client.get(self.challenges_url)
            resp.raise_for_status()
            return resp.text

    def discover(self) -> Iterator[RawChallenge]:
        entries = yaml.safe_load(self._fetch()) or []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            difficulty = entry.get("difficulty")
            yield RawChallenge(
                origin=Origin.JUICESHOP,
                title=entry["name"],
                event_name=EVENT_NAME,
                raw_category=entry.get("category"),
                difficulty=str(difficulty) if difficulty is not None else None,
                tags=_normalize_tags(entry.get("tags")),
                description=entry.get("description"),
                writeups=_build_writeups(entry),
                source=Source(
                    origin=Origin.JUICESHOP,
                    url="https://github.com/juice-shop/juice-shop",
                    repo="juice-shop/juice-shop",
                    is_official=True,
                    license=licenses.from_spdx("MIT", note="Juice Shop app is MIT"),
                ),
                # Preserve the stable per-challenge key for later dedup/traceability.
                extra={"key": entry.get("key")},
            )


CONNECTOR = JuiceShopConnector
