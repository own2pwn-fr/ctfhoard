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

#: Filename prefixes (lowercased) of root license files we probe when a repo declares
#: no SPDX. Matched case-insensitively against the *start* of a file name so common
#: variants — ``LICENSE``, ``LICENSE.md``, ``LICENSE-MIT``, ``LICENCE``, ``COPYING``,
#: ``COPYING.md``, ``UNLICENSE`` — are all picked up.
_LICENSE_PREFIXES: tuple[str, ...] = ("license", "licence", "copying", "unlicense")

#: Upper bound on bytes read from an untrusted LICENSE probe. A repo can ship a
#: multi-gigabyte file named ``LICENSE`` (or anything matching the prefixes); slurping
#: it whole would OOM a cheap metadata probe, so we only read a bounded prefix — the
#: license marker phrases all live in the first few hundred bytes anyway.
_MAX_PROBE_BYTES: int = 1024 * 1024  # 1 MiB

#: Default wall-clock timeout (seconds) for a single git subprocess. Generous enough
#: for a large shallow clone but finite, so a huge or stalled repo cannot hang the
#: whole run forever — on timeout the process is killed and the repo is skipped.
_GIT_TIMEOUT: float = 600.0


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


def _read_text_bounded(path: Path, *, limit: int = _MAX_PROBE_BYTES) -> str:
    """Read at most ``limit`` bytes of ``path`` and decode as text.

    Untrusted probe files (LICENSE variants) can be arbitrarily large; reading a
    bounded prefix keeps a cheap metadata probe from OOM-ing on a pathological file.
    """
    with path.open("rb") as fh:
        raw = fh.read(limit)
    return raw.decode("utf-8", errors="replace")


def _run_git(
    args: list[str], *, cwd: Path | None = None, timeout: float = _GIT_TIMEOUT
) -> str:
    """Run a git subcommand, returning stdout; raise :class:`MirrorError` on failure.

    ``timeout`` bounds the wall-clock time of the call; on expiry the child process is
    killed (by :func:`subprocess.run`) and a :class:`MirrorError` is raised so the
    caller can skip the offending repo and continue instead of hanging forever.
    """
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            env=_git_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run has already killed and reaped the child on timeout.
        cmd = " ".join(["git", *args])
        raise MirrorError(f"`{cmd}` timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0:
        cmd = " ".join(["git", *args])
        raise MirrorError(f"`{cmd}` failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def clone_pinned(
    repo: str, workdir: Path, *, depth: int = 1, timeout: float = _GIT_TIMEOUT
) -> tuple[Path, str]:
    """Shallow-clone ``owner/name`` into ``workdir`` and return (path, commit_sha).

    The clone lands in ``workdir/<owner>__<name>``. If that directory already holds
    a git checkout it is reused as-is (idempotent / resumable — re-running the
    connector does not re-download), otherwise a fresh ``git clone --depth`` is run.
    ``commit_sha`` is the resolved ``HEAD`` (``git rev-parse HEAD``), i.e. the exact
    commit the emitted challenges are pinned to for provenance. ``timeout`` bounds
    each underlying git call so a huge or stalled clone raises :class:`MirrorError`
    instead of hanging the run.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    local_path = workdir / _repo_dirname(repo)
    if not (local_path / ".git").exists():
        url = f"https://github.com/{repo}.git"
        _run_git(["clone", "--depth", str(depth), url, str(local_path)], timeout=timeout)
    sha = _run_git(["rev-parse", "HEAD"], cwd=local_path, timeout=timeout).strip()
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
    best: LicenseInfo | None = None
    for candidate in _iter_license_files(local_path):
        text = _read_text_bounded(candidate)
        info = licenses.detect_from_text(text, source_file=candidate.name)
        if best is None or info.confidence > best.confidence:
            best = info
    if best is not None:
        return best
    return LicenseInfo(
        redistributable=False,
        confidence=0.0,
        note="no declared SPDX and no root LICENSE/COPYING file found",
    )


def _iter_license_files(local_path: Path) -> list[Path]:
    """Return root-level LICENSE/LICENCE/COPYING/UNLICENSE files, case-insensitively.

    Globbing on a case-sensitive filesystem would miss ``licence``/``Copying`` etc.,
    so we scan the root directory once and prefix-match each entry's lowercased name
    against :data:`_LICENSE_PREFIXES` (which also catches variants like ``LICENSE-MIT``
    and ``COPYING.md``). Returned sorted for deterministic probing order.
    """
    try:
        entries = sorted(local_path.iterdir())
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if entry.is_file() and entry.name.lower().startswith(_LICENSE_PREFIXES)
    ]
