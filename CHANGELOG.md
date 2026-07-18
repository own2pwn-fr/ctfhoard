# Changelog

## [Unreleased]

### Security & hardening (adversarial multi-agent review, 26 findings fixed)
- **SSRF guard** (`netguard.py`): writeup fetches are validated against a public-IP
  allowlist (reject loopback/link-local/private/reserved, IPv4-mapped IPv6),
  redirects re-validated per hop, bodies streamed with a 25 MiB cap — closes an
  SSRF/exfil hole where an attacker-supplied writeup URL could read cloud-metadata
  and get published to the public dataset.
- **No silent data loss**: Unicode-aware `slugify` + raw-title hash anchoring stops
  non-ASCII (CJK/Cyrillic) titles from colliding into one id; `content_fingerprint`
  now hashes large (LFS) source files so distinct challenges no longer merge.
- **No silent drops at scale**: transient 5xx/network errors are retried+skipped
  (not fatal) in discovery and the Hackropole/CTFtime crawls; CTFtime window
  pagination keeps boundary-second events.
- **Robustness**: `git clone` timeout, discovered-repo size cap, bounded reads of
  untrusted LICENSE/challenge.json, atomic `.part` downloads, license-filename
  variants, repo-root treated as a challenge leaf, relative `corpus_path` (no
  absolute-path leak), writeup-orphan pruning, symlink-escape refusal.
- 41 new regression tests (101 total).

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
- GitHub-wide discovery (`discover.py` + `discover-github` CLI): sharded repo
  search with recursive `created:`/`stars:` bisection to beat the 1000-result
  cap (verified: `topic:ctf-writeups` alone = 1819 repos), writing
  `data/discovered_repos.jsonl` that `git_repo` ingests via `discovered_path`.
