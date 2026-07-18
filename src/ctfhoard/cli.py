"""``ctfhoard`` command-line interface.

Wires the pipeline together: a connector discovers raw challenges, ``normalize``
canonicalizes them, ``dedup`` collapses mirrors of the same challenge, and
``storage`` writes the JSONL/Parquet catalog. Each source is one connector; the CLI
just orchestrates.
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from ctfhoard import hf
from ctfhoard.connectors import load_registry
from ctfhoard.corpus import materialize_challenge
from ctfhoard.dedup import cluster_by_fingerprint, merge_cluster
from ctfhoard.http import PoliteClient, make_client
from ctfhoard.normalize import normalize
from ctfhoard.ratelimit import RateLimiter
from ctfhoard.schema import Challenge
from ctfhoard.storage import CatalogWriter, read_jsonl

app = typer.Typer(help="Aggregate whitebox CTF sources and writeups into one corpus.")

DEFAULT_DATA = Path("data")


@app.command("list-connectors")
def list_connectors() -> None:
    """List every available source connector."""
    registry = load_registry()
    if not registry:
        typer.echo("no connectors available")
        raise typer.Exit(code=0)
    for name, cls in sorted(registry.items()):
        typer.echo(f"{name:16} {cls.__module__}")


def _dedup(challenges: list[Challenge]) -> list[Challenge]:
    """Collapse fingerprint-identical mirrors into canonical records."""
    clusters = cluster_by_fingerprint(challenges)
    canonical: list[Challenge] = []
    for members in clusters.values():
        merged = merge_cluster(members)
        for m in members:
            if m.id != merged.id:
                m.duplicate_of = merged.id
        canonical.append(merged)
    return canonical


@app.command()
def ingest(
    connector: str = typer.Argument(..., help="Connector name (see list-connectors)."),
    data_dir: Path = typer.Option(DEFAULT_DATA, help="Root data directory."),
    parquet: bool = typer.Option(False, help="Also write a Parquet export."),
    no_dedup: bool = typer.Option(False, help="Skip fingerprint deduplication."),
    no_materialize: bool = typer.Option(
        False, help="Skip writing hard copies of sources/writeups into data/corpus/."
    ),
    no_fetch_writeups: bool = typer.Option(
        False, help="Do not follow external writeup links to store their content."
    ),
    writeup_delay: float = typer.Option(
        1.0, help="Min seconds between external writeup fetches (be polite)."
    ),
) -> None:
    """Run one connector end-to-end and write its catalog shard."""
    registry = load_registry()
    cls = registry.get(connector)
    if cls is None:
        typer.echo(f"unknown connector '{connector}'. Available: {', '.join(registry)}")
        raise typer.Exit(code=1)

    workdir = data_dir / "raw" / connector
    instance = cls(workdir=workdir)

    logger.info("ingesting via connector '{}'", connector)
    challenges: list[Challenge] = []
    for raw in instance.discover():
        challenges.append(normalize(raw))
    logger.info("discovered {} raw challenges", len(challenges))

    if not no_dedup:
        before = len(challenges)
        challenges = _dedup(challenges)
        logger.info("dedup collapsed {} -> {}", before, len(challenges))

    if not no_materialize:
        corpus_root = data_dir / "corpus"
        # Resolve the data dir so corpus_path is stored RELATIVE to it (e.g.
        # 'corpus/github/...') even when --data-dir is an absolute path — an
        # absolute machine path must never leak into the published catalog.
        repo_root = data_dir.resolve()
        client = (
            None
            if no_fetch_writeups
            else PoliteClient(make_client(), RateLimiter(min_interval=writeup_delay))
        )
        try:
            for ch in challenges:
                raw_dir = (
                    Path(ch.corpus_path)
                    if ch.corpus_path and Path(ch.corpus_path).exists()
                    else None
                )
                materialize_challenge(
                    ch, corpus_root, raw_dir=raw_dir, client=client, repo_root=repo_root
                )
        finally:
            if client is not None:
                client.close()
        logger.info("materialized hard copies under {}", corpus_root)

    writer = CatalogWriter(data_dir / "catalog" / connector)
    writer.extend(challenges)
    jsonl = writer.write_jsonl()
    logger.info("wrote {} records to {}", len(writer), jsonl)
    if parquet:
        pq = writer.write_parquet()
        logger.info("parquet: {}", pq or "skipped (pyarrow missing)")


@app.command("discover-github")
def discover_github(
    max_repos: int = typer.Option(
        None, help="Cap the number of NEW repos discovered (default: no cap)."
    ),
    out: Path = typer.Option(
        DEFAULT_DATA / "discovered_repos.jsonl",
        help="Output JSONL path (consumed by the git_repo connector).",
    ),
    token: str = typer.Option(
        None,
        help="GitHub token; defaults to GH_TOKEN/GITHUB_TOKEN env or `gh auth token`.",
    ),
) -> None:
    """Discover CTF challenge-source and writeup repos on GitHub into a JSONL list.

    This is DISCOVERY only — it finds repositories, it does not extract challenges.
    Feed the resulting file to the ``git_repo`` connector (``--discovered-path``) to
    mirror and walk them.
    """
    from ctfhoard.discover import discover_all, resolve_token, write_discovered

    resolved = resolve_token(token)
    if not resolved:
        logger.warning("no GitHub token found; unauthenticated search is heavily limited")

    found = discover_all(token=resolved, max_repos=max_repos)
    path = write_discovered(found.values(), out)

    by_kind: dict[str, int] = {}
    for cand in found.values():
        by_kind[cand.kind] = by_kind.get(cand.kind, 0) + 1
    logger.info("discovered {} repos -> {}", len(found), path)
    typer.echo(f"discovered {len(found)} repos -> {path}")
    for kind, n in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {kind:10} {n}")


@app.command()
def stats(data_dir: Path = typer.Option(DEFAULT_DATA, help="Root data directory.")) -> None:
    """Summarize the aggregate catalog across all connector shards."""
    catalog_root = data_dir / "catalog"
    shards = list(catalog_root.rglob("*.jsonl"))
    if not shards:
        typer.echo("no catalog shards found")
        raise typer.Exit(code=0)

    total = whitebox = redistributable = with_writeups = 0
    by_category: dict[str, int] = {}
    for shard in shards:
        for ch in read_jsonl(shard):
            total += 1
            whitebox += ch.has_source
            redistributable += ch.redistributable
            with_writeups += bool(ch.writeups)
            by_category[ch.category.value] = by_category.get(ch.category.value, 0) + 1

    typer.echo(f"challenges:      {total}")
    typer.echo(f"  whitebox:      {whitebox}")
    typer.echo(f"  redistributable:{redistributable}")
    typer.echo(f"  with writeups: {with_writeups}")
    typer.echo("by category:")
    for cat, n in sorted(by_category.items(), key=lambda kv: -kv[1]):
        typer.echo(f"  {cat:12} {n}")


@app.command("publish-hf")
def publish_hf(
    data_dir: Path = typer.Option(DEFAULT_DATA, help="Root data directory."),
    dry_run: bool = typer.Option(
        False, help="Report what would be uploaded without touching the network."
    ),
    corpus_only: bool = typer.Option(False, help="Publish only the corpus (skip catalog)."),
    catalog_only: bool = typer.Option(False, help="Publish only the catalog (skip corpus)."),
) -> None:
    """Publish the local corpus and catalog to the Hugging Face dataset repo."""
    if corpus_only and catalog_only:
        typer.echo("--corpus-only and --catalog-only are mutually exclusive")
        raise typer.Exit(code=1)

    if not catalog_only:
        corpus_dir = data_dir / "corpus"
        stats = hf.corpus_stats(corpus_dir)
        logger.info(
            "corpus: {} files, {} bytes ({})",
            stats["files"],
            stats["total_bytes"],
            hf.HF_CORPUS_DATASET,
        )
        result = hf.publish_corpus(corpus_dir, dry_run=dry_run)
        logger.info("corpus publish: {}", result)
        typer.echo(result)

    if not corpus_only:
        catalog_dir = data_dir / "catalog"
        result = hf.publish_catalog(catalog_dir, dry_run=dry_run)
        logger.info("catalog publish: {}", result)
        typer.echo(result)


if __name__ == "__main__":
    app()
