"""Catalog persistence.

The normalized catalog is the machine-readable heart of the corpus: one
:class:`~ctfhoard.schema.Challenge` per line of JSONL (diff-friendly, git-trackable,
streamable), with an optional Parquet export for analytics/ML consumers. The heavy
mirrored artifacts themselves live under ``data/corpus/`` (Git LFS); the catalog
only references them by ``corpus_path`` + per-file hashes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from ctfhoard.schema import Challenge


class CatalogWriter:
    """Append/collect challenges and flush them to JSONL (and optionally Parquet)."""

    def __init__(self, catalog_dir: Path) -> None:
        self.catalog_dir = catalog_dir
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self._records: list[Challenge] = []

    def add(self, challenge: Challenge) -> None:
        self._records.append(challenge)

    def extend(self, challenges: Iterable[Challenge]) -> None:
        self._records.extend(challenges)

    def __len__(self) -> int:
        return len(self._records)

    def write_jsonl(self, filename: str = "challenges.jsonl") -> Path:
        """Write all buffered records as JSON Lines. Returns the path written."""
        out = self.catalog_dir / filename
        with out.open("w", encoding="utf-8") as fh:
            for ch in self._records:
                fh.write(ch.model_dump_json(exclude_none=True))
                fh.write("\n")
        return out

    def write_parquet(self, filename: str = "challenges.parquet") -> Path | None:
        """Write a flat Parquet export. Requires the ``parquet`` extra; returns None
        (and does nothing) if pyarrow is unavailable so this stays optional."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return None
        rows = [ch.model_dump(mode="json") for ch in self._records]
        if not rows:
            return None
        out = self.catalog_dir / filename
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out)
        return out


def read_jsonl(path: Path) -> Iterator[Challenge]:
    """Stream a JSONL catalog back into :class:`Challenge` objects."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Challenge.model_validate_json(line)


def merge_jsonl(paths: Iterable[Path]) -> dict[str, Challenge]:
    """Load several JSONL shards, upserting by ``Challenge.id`` (last write wins)."""
    merged: dict[str, Challenge] = {}
    for p in paths:
        for ch in read_jsonl(p):
            merged[ch.id] = ch
    return merged
