"""Connector contract.

A connector knows how to talk to exactly one kind of source and yields
:class:`~ctfhoard.schema.RawChallenge` objects. It does NOT normalize, dedup, or
write the catalog — that is the pipeline's job. Keeping connectors to a single
responsibility (extract from source X) makes them small, independently testable,
and safe to implement in parallel.

Implement a new source by subclassing :class:`Connector` and yielding
``RawChallenge`` from :meth:`discover`. If the source materializes challenge files
on disk, set ``RawChallenge.local_dir`` and let the pipeline build the file
manifest; otherwise attach :class:`~ctfhoard.schema.FileEntry` items directly.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path

from ctfhoard.schema import Origin, RawChallenge


class Connector(abc.ABC):
    """Base class every source connector implements."""

    #: The provenance identity every RawChallenge from this connector carries.
    origin: Origin

    #: Short stable name used on the CLI (e.g. 'juiceshop'). Class attribute so the
    #: registry can index connectors without instantiating them.
    cli_name: str = "connector"

    def __init__(self, workdir: Path) -> None:
        """`workdir` is a scratch dir the connector may use for clones/downloads."""
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        """Instance-level accessor for :attr:`cli_name`."""
        return self.cli_name

    @abc.abstractmethod
    def discover(self) -> Iterator[RawChallenge]:
        """Yield every challenge this source exposes.

        Must be resumable/idempotent where feasible: re-running should not corrupt
        state, and should reuse cached downloads under ``self.workdir``.
        """
