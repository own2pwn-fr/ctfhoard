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

from ctfhoard.connectors import load_registry
from ctfhoard.dedup import cluster_by_fingerprint, merge_cluster
from ctfhoard.normalize import normalize
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

    writer = CatalogWriter(data_dir / "catalog" / connector)
    writer.extend(challenges)
    jsonl = writer.write_jsonl()
    logger.info("wrote {} records to {}", len(writer), jsonl)
    if parquet:
        pq = writer.write_parquet()
        logger.info("parquet: {}", pq or "skipped (pyarrow missing)")


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


if __name__ == "__main__":
    app()
