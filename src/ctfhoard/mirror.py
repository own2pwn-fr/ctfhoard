"""Reusable Git mirroring helpers for repo-based connectors.

Connectors that ingest challenge *sources* from GitHub (the ``git_repo`` connector
and any future repo-backed one) all need the same two primitives: pin a repo at an
exact commit on disk, and figure out what license that repo ships under. Both live
here so the connectors stay small and the mirroring policy (shallow clones, LFS
pointers left un-smudged, per-repo license detection) is defined in one place.

We shell out to the ``git`` CLI rather than pulling in a git library: it is already
required in the environment, handles auth/redirects/partial-clone natively, and
keeps this module dependency-light (stdlib + our :mod:`ctfhoard.licenses`).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ctfhoard import licenses
from ctfhoard.schema import LicenseInfo

#: Root license filenames we probe, in priority order, when a repo declares no SPDX.
_LICENSE_FILENAMES: tuple[str, ...] = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "COPYING.txt",
    "COPYING.md",
)


class MirrorError(RuntimeError):
    """Raised when a git mirror operation (clone / rev-parse) fails."""


def _repo_dirname(repo: str) -> str:
    """Flatten ``owner/name`` into a single filesystem-safe clone dir name."""
    owner, _, name = repo.partition("/")
    return f"{owner}__{name}" if name else owner


def _git_env() -> dict[str, str]:
    """Environment for git calls: never smudge LFS blobs, never prompt for creds."""
    env = dict(os.environ)
    # Keep Git LFS pointer files as-is: large binaries stay as ~130-byte pointers
    # instead of blowing up the clone (and our disk) with gigabytes of media.
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    # Fail fast instead of hanging on an interactive credential prompt in CI/agents.
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    """Run a git subcommand, returning stdout; raise :class:`MirrorError` on failure."""
    proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        env=_git_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        cmd = " ".join(["git", *args])
        raise MirrorError(f"`{cmd}` failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def clone_pinned(repo: str, workdir: Path, *, depth: int = 1) -> tuple[Path, str]:
    """Shallow-clone ``owner/name`` into ``workdir`` and return (path, commit_sha).

    The clone lands in ``workdir/<owner>__<name>``. If that directory already holds
    a git checkout it is reused as-is (idempotent / resumable — re-running the
    connector does not re-download), otherwise a fresh ``git clone --depth`` is run.
    ``commit_sha`` is the resolved ``HEAD`` (``git rev-parse HEAD``), i.e. the exact
    commit the emitted challenges are pinned to for provenance.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    local_path = workdir / _repo_dirname(repo)
    if not (local_path / ".git").exists():
        url = f"https://github.com/{repo}.git"
        _run_git(["clone", "--depth", str(depth), url, str(local_path)])
    sha = _run_git(["rev-parse", "HEAD"], cwd=local_path).strip()
    if not sha:
        raise MirrorError(f"could not resolve HEAD for {repo} at {local_path}")
    return local_path, sha


def detect_repo_license(local_path: Path, *, declared_spdx: str | None) -> LicenseInfo:
    """Resolve the license for a cloned repo.

    Precedence:

    1. ``declared_spdx`` (the blanket SPDX id from ``seeds/official_repos.yaml``) is
       authoritative when present — we trust the curated seed over text heuristics.
    2. Otherwise probe a root ``LICENSE``/``COPYING`` file and run text detection
       (:func:`ctfhoard.licenses.detect_from_text`), recording which file it came
       from in ``source_file``.
    3. If nothing is found, return a conservative non-redistributable
       :class:`~ctfhoard.schema.LicenseInfo` (all-rights-reserved by default).
    """
    if declared_spdx:
        return licenses.from_spdx(declared_spdx, note="declared in seeds")
    for name in _LICENSE_FILENAMES:
        candidate = local_path / name
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="replace")
            return licenses.detect_from_text(text, source_file=name)
    return LicenseInfo(
        redistributable=False,
        confidence=0.0,
        note="no declared SPDX and no root LICENSE/COPYING file found",
    )
