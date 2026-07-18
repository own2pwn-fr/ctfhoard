# Changelog

## [Unreleased]

### Added
- Project scaffold: normalized `Challenge` schema (the data contract), connector
  interface, and the ingest → normalize → dedup → catalog pipeline.
- Core engine: license/SPDX detection with a conservative `redistributable` flag,
  content-fingerprint deduplication (Merkle hash over the source-file set),
  JSONL/Parquet catalog storage, polite rate limiting (token bucket + crawl-delay),
  and a shared HTTP client.
- CLI (`ctfhoard`): `ingest`, `list-connectors`, `stats`.
- Connectors: `juiceshop` (challenges.yml, MIT), `hackropole` (FCSC/ANSSI static
  site, etalab-2.0), `git_repo` (generic walker over official/archive repos from
  `seeds/official_repos.yaml`, per-repo license detection, SHA-pinned provenance),
  `ctftime` (JSON API metadata graph + polite HTML writeup crawl, 10s crawl-delay).
- `mirror.py`: SHA-pinned shallow clones + repo license detection.
- Hard-copy corpus: `corpus.py` materializes source files AND writeup content
  (following external links) into `data/corpus/`.
- Hybrid storage: GitHub holds code + catalog + text corpus; the complete
  hard-copy archive (incl. binaries) is published to the Hugging Face dataset
  `own2pwn-fr/ctfhoard-corpus` via `ctfhoard publish-hf` (`hf.py`). Binaries are
  gitignored on GitHub (kept off the 1 GB free-LFS wall).
