# ctfhoard

> Aggregate **all whitebox CTF challenge sources and writeups** into one normalized,
> provenance-tracked, deduplicated corpus — from CTFtime, GitHub, Hackropole,
> Juice Shop, and more.

`ctfhoard` scrapes and mirrors publicly available Capture-The-Flag material and
folds it into a single machine- and human-readable catalog. Every challenge record
is normalized to one schema, carries full provenance (which repo, which commit) and
a detected license, and is deduplicated by content so the same challenge mirrored in
many places becomes one canonical record with an attribution graph.

## Why

Whitebox CTF challenges (those shipping their source) are a huge, scattered corpus
of real-world security exercises. They live across CTFtime writeup links, hundreds
of GitHub repos, and a handful of clean official archives. `ctfhoard` unifies them
so they can be browsed, searched, and studied — and, later, used to build datasets.

## Architecture

- **`schema.py`** — the single normalized contract (`Challenge`, `Source`,
  `Writeup`, `LicenseInfo`, `FileEntry`). Everything produces and consumes this.
- **`connectors/`** — one module per source. Each yields loose `RawChallenge`
  objects from its source; it does nothing else.
- **`normalize.py`** — turns a `RawChallenge` into a canonical `Challenge`
  (stable id, category mapping, file manifest, content fingerprint).
- **`dedup.py`** — Merkle content fingerprint over the source-file set; collapses
  mirrors of the same challenge into one canonical record.
- **`licenses.py`** — SPDX detection → conservative `redistributable` flag.
- **`mirror.py`** — pin-by-SHA tarball / `git --mirror` fetching with LFS handling.
- **`storage.py`** — JSONL (+ optional Parquet) catalog; heavy artifacts under
  `data/corpus/` via Git LFS.
- **`cli.py`** — `ctfhoard ingest <connector>`, `list-connectors`, `stats`.

## Usage

```bash
uv venv && uv pip install -e '.[dev,parquet,dedup]'
ctfhoard list-connectors
ctfhoard ingest juiceshop        # ingest one source → data/catalog/juiceshop/
ctfhoard stats                   # summarize the aggregate catalog
```

## Licensing & ethics

The corpus mirrors third-party content under **heterogeneous and often absent**
licenses. `ctfhoard` records a license and a conservative `redistributable` flag per
challenge (defaulting to *not* redistributable when no license is detected), so the
catalog can always be filtered down to provably redistributable material.
Source-site etiquette (robots `Crawl-delay`, `ai-train` signals) is honored by the
connectors. Challenge and writeup copyrights remain with their original authors.

Licensed under **AGPL-3.0-or-later**.
