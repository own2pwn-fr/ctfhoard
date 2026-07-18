"""Connector registry.

Each connector module exposes a module-level ``CONNECTOR`` attribute bound to its
:class:`~ctfhoard.connectors.base.Connector` subclass. We import them defensively so
a not-yet-implemented (or optional-dependency-missing) connector never breaks the
whole CLI — it is simply absent from the registry.
"""

from __future__ import annotations

import importlib

from ctfhoard.connectors.base import Connector

# module name (under ctfhoard.connectors) -> friendly CLI name is derived from the
# connector's own `.name`. Add new connectors here.
# Only real connector modules. Sources like sajjadium / pwncollege / NYU-bench are
# ingested through `git_repo` (via seeds/official_repos.yaml), not as separate
# connectors; GitHub-wide discovery lives in `ctfhoard.discover` (a module + the
# `discover-github` CLI command), not here.
_MODULES = [
    "juiceshop",
    "hackropole",
    "git_repo",
    "ctftime",
]


def load_registry() -> dict[str, type[Connector]]:
    """Return {connector_name: connector_class} for every importable connector."""
    registry: dict[str, type[Connector]] = {}
    for mod_name in _MODULES:
        try:
            module = importlib.import_module(f"ctfhoard.connectors.{mod_name}")
        except Exception:  # noqa: BLE001 — missing optional deps must not break others
            continue
        connector_cls = getattr(module, "CONNECTOR", None)
        if connector_cls is None or not issubclass(connector_cls, Connector):
            continue
        # instantiate-free name: read the class attribute if present, else module name
        name = getattr(connector_cls, "cli_name", mod_name)
        registry[name] = connector_cls
    return registry
