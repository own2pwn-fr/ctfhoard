"""License detection and redistribution decisions.

The corpus mirrors code from many repos under heterogeneous — and frequently
absent — licenses. Rather than guess at redistribution rights globally, we detect
a license per source and derive a conservative ``redistributable`` flag. Default is
False (all-rights-reserved) whenever nothing is detected, so an operator can always
filter the catalog down to what is provably safe to redistribute.
"""

from __future__ import annotations

import re

from ctfhoard.schema import LicenseInfo

# SPDX ids we consider redistributable in an open aggregate (permissive + copyleft;
# copyleft is redistributable, it just constrains how). Anything not here — or not
# detected at all — is treated as non-redistributable until proven otherwise.
REDISTRIBUTABLE_SPDX: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "MPL-2.0",
        "GPL-2.0-only",
        "GPL-2.0-or-later",
        "GPL-3.0-only",
        "GPL-3.0-or-later",
        "LGPL-2.1-or-later",
        "LGPL-3.0-or-later",
        "AGPL-3.0-or-later",
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "Unlicense",
        "WTFPL",
        "0BSD",
        "etalab-2.0",  # Licence Ouverte 2.0 (Hackropole/ANSSI) — open with attribution
    }
)

# Marker phrases → SPDX id, for detecting a license from a LICENSE file's text when
# an authoritative API id is unavailable. Order matters: more specific first.
_MARKER_RULES: list[tuple[str, str]] = [
    (r"GNU AFFERO GENERAL PUBLIC LICENSE.*Version 3", "AGPL-3.0-or-later"),
    (r"GNU GENERAL PUBLIC LICENSE.*Version 3", "GPL-3.0-or-later"),
    (r"GNU GENERAL PUBLIC LICENSE.*Version 2", "GPL-2.0-or-later"),
    (r"GNU LESSER GENERAL PUBLIC LICENSE.*Version 3", "LGPL-3.0-or-later"),
    (r"Apache License.*Version 2\.0", "Apache-2.0"),
    (r"Mozilla Public License Version 2\.0", "MPL-2.0"),
    (r"Permission is hereby granted, free of charge.*MIT", "MIT"),
    (r'Permission is hereby granted, free of charge, to any person obtaining a copy',
     "MIT"),  # canonical MIT opening even without the word "MIT"
    (r"Redistribution and use in source and binary forms.*neither the name",
     "BSD-3-Clause"),
    (r"Redistribution and use in source and binary forms", "BSD-2-Clause"),
    (r"This is free and unencumbered software released into the public domain",
     "Unlicense"),
    (r"DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE", "WTFPL"),
    (r"Licence Ouverte|Open Licence|etalab", "etalab-2.0"),
    (r"Creative Commons Attribution-ShareAlike 4\.0", "CC-BY-SA-4.0"),
    (r"Creative Commons Attribution 4\.0", "CC-BY-4.0"),
    (r"CC0 1\.0 Universal", "CC0-1.0"),
]

_COMPILED_MARKERS = [(re.compile(p, re.IGNORECASE | re.DOTALL), spdx) for p, spdx in _MARKER_RULES]


def is_redistributable(spdx_id: str | None) -> bool:
    """Conservative redistribution decision from an SPDX id."""
    if not spdx_id:
        return False
    return spdx_id in REDISTRIBUTABLE_SPDX


def from_spdx(
    spdx_id: str | None, *, confidence: float = 1.0, note: str | None = None
) -> LicenseInfo:
    """Build a :class:`LicenseInfo` from a known SPDX id (e.g. GitHub license API)."""
    if not spdx_id or spdx_id.upper() in {"NOASSERTION", "NONE"}:
        return LicenseInfo(redistributable=False, confidence=0.0, note=note)
    return LicenseInfo(
        spdx_id=spdx_id,
        confidence=confidence,
        redistributable=is_redistributable(spdx_id),
        note=note,
    )


def detect_from_text(text: str, *, source_file: str | None = None) -> LicenseInfo:
    """Best-effort SPDX detection from raw LICENSE/COPYING text."""
    if not text or not text.strip():
        return LicenseInfo(redistributable=False, confidence=0.0)
    head = text[:4000]
    for pattern, spdx in _COMPILED_MARKERS:
        if pattern.search(head):
            return LicenseInfo(
                spdx_id=spdx,
                confidence=0.75,  # text heuristics are strong but not authoritative
                redistributable=is_redistributable(spdx),
                source_file=source_file,
            )
    return LicenseInfo(
        redistributable=False,
        confidence=0.0,
        source_file=source_file,
        note="license file present but unrecognized",
    )
