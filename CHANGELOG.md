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
