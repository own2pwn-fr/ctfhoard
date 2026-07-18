"""Generic GitHub challenge-repo connector.

Most authoritative CTF *source* lives in a handful of big GitHub repos: official
organizer archives (google/google-ctf, Nautilus-Institute/quals-*), broad community
archives (sajjadium/ctf-archives, pwncollege/ctf-archive), and benchmark repos with
clean metadata (NYU-LLM-CTF/NYU_CTF_Bench). This one connector mirrors them all: for
each repo listed in ``seeds/official_repos.yaml`` it pins the repo at a commit
(:mod:`ctfhoard.mirror`), detects its license, then walks the tree emitting one
:class:`~ctfhoard.schema.RawChallenge` per *challenge directory* with full provenance
(repo + commit SHA + sub-path).

Cross-repo deduplication (the same challenge appearing in an official repo *and*
sajjadium *and* a team fork) is handled downstream by the engine via content
fingerprints (:mod:`ctfhoard.dedup`). This connector's only job is faithful,
exhaustive extraction — it does not merge or judge, it just mirrors.

Challenge-directory heuristic (see :func:`_iter_challenge_dirs`)
--------------------------------------------------------------
Walk top-down and stop at the *most specific* directory that holds artifacts:

* A directory containing a ``challenge.json`` (NYU-bench style) is a single
  challenge — the cleanest signal; we parse it for name/category/flag and stop.
* Otherwise a directory that *directly* contains challenge artifacts — a
  ``Dockerfile``/``docker-compose*.yml``, source files (``*.py,*.c,*.cpp,*.rs,
  *.go,*.js,*.php,*.sol,*.S``), or a handout (``*.tar.gz``/``*.zip``) — is treated
  as one challenge leaf; its own sub-dirs (``src/``, ``public/``, ``solution/``)
  are its components and are *not* descended into. This picks the leaf that owns
  the artifacts and never emits both a parent and its children.
* A directory with no direct artifacts is a container (event/year/category) and we
  recurse into it. Vendor/noise dirs (``.git``, ``node_modules``, ``.github`` …)
  are skipped, and asset-only dirs (images with no source) simply yield nothing.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ctfhoard.connectors.base import Connector
from ctfhoard.mirror import clone_pinned, detect_repo_license
from ctfhoard.normalize import map_category
from ctfhoard.schema import Category, LicenseInfo, Origin, RawChallenge, Source

#: Default location of the seed file (repo-root ``seeds/official_repos.yaml``).
#: git_repo.py -> connectors -> ctfhoard -> src -> <repo root>.
_DEFAULT_SEEDS = Path(__file__).resolve().parents[3] / "seeds" / "official_repos.yaml"

#: Seed sections flattened, in authority order, into one repo list.
_SEED_SECTIONS = ("official_sources", "community_archives", "writeup_archives")

#: Source-code extensions that mark a directory as holding challenge source.
_SOURCE_EXTS = frozenset(
    {".py", ".c", ".cpp", ".cc", ".rs", ".go", ".js", ".php", ".sol", ".s"}
)
#: Packaged-handout extensions (distributed challenge artifacts).
_HANDOUT_EXTS = frozenset({".gz", ".tgz", ".zip", ".xz", ".bz2", ".7z"})
#: Docker artifact file names (compose variants matched by prefix separately).
_DOCKER_NAMES = frozenset({"dockerfile"})

#: Component sub-directory names: a *part* of a challenge (handout vs. deploy vs.
#: solution), not a sub-challenge. When a directory's immediate sub-dirs are ALL of
#: these, the directory itself is the challenge *root* and its sub-dirs are its
#: components — the layout many archives use (justCTF/pwn.college ``public``+
#: ``private``) where nothing sits at the challenge dir's own top level, so
#: artifact-presence alone would under-segment it into its parts.
_COMPONENT_NAMES = frozenset(
    {
        "public",
        "private",
        "src",
        "source",
        "sources",
        "dist",
        "handout",
        "handouts",
        "deploy",
        "deployment",
        "solution",
        "solutions",
        "solve",
        "solver",
        "writeup",
        "writeups",
        "exploit",
        "exploits",
        "poc",
        "setup",
        "service",
        "server",
        "client",
        "app",
        "bin",
        "build",
        "files",
        "attachments",
        "share",
        "upload",
        "uploads",
        "release",
        "docker",
    }
)

#: Generic wrapper segments that are never a meaningful event name on their own.
_WRAPPER_SEGMENTS = frozenset({"ctfs", "challenges", "challenge", "chall", "chals", "tasks"})

#: Directory names never worth descending into.
_NOISE_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".gitlab",
        "node_modules",
        "__pycache__",
        ".idea",
        ".vscode",
        "venv",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
    }
)

#: Path segments that denote an edition rather than an event/category/challenge.
_EDITION_TOKENS = frozenset({"quals", "qualifiers", "finals", "final", "onsite"})

#: owner (lowercased) / repo-specific -> a more precise Origin than plain GITHUB.
_ORIGIN_BY_REPO: dict[str, Origin] = {
    "google/google-ctf": Origin.GOOGLE_CTF,
}
_ORIGIN_BY_OWNER: dict[str, Origin] = {
    "google": Origin.GOOGLE_CTF,
    "sajjadium": Origin.SAJJADIUM,
    "pwncollege": Origin.PWNCOLLEGE,
    "nyu-llm-ctf": Origin.NYU_CTF_BENCH,
}


def _origin_for_repo(repo: str) -> Origin:
    """Pick the most specific :class:`Origin` for a repo, else generic GitHub."""
    if repo in _ORIGIN_BY_REPO:
        return _ORIGIN_BY_REPO[repo]
    owner = repo.split("/", 1)[0].lower()
    return _ORIGIN_BY_OWNER.get(owner, Origin.GITHUB)


def _is_noise(name: str) -> bool:
    return name in _NOISE_DIRS or name.startswith(".git")


def _is_docker_file(name: str) -> bool:
    low = name.lower()
    return low in _DOCKER_NAMES or low.startswith("docker-compose") or low == "compose.yaml"


def _is_source_name(name: str) -> bool:
    return Path(name).suffix.lower() in _SOURCE_EXTS


def _is_handout_name(name: str) -> bool:
    low = name.lower()
    return low.endswith(".tar.gz") or Path(low).suffix in _HANDOUT_EXTS


def _is_artifact_name(name: str) -> bool:
    return _is_docker_file(name) or _is_source_name(name) or _is_handout_name(name)


def _has_direct_artifacts(d: Path) -> bool:
    """True if ``d`` directly (non-recursively) contains challenge artifacts."""
    return any(c.is_file() and _is_artifact_name(c.name) for c in d.iterdir())


def _subtree_has_artifacts(d: Path) -> bool:
    """True if any challenge artifact exists anywhere in ``d``'s subtree.

    Noise dirs (``.git``/``node_modules``/…) are pruned from the walk.
    """
    for _dirpath, dirnames, filenames in os.walk(d):
        dirnames[:] = [n for n in dirnames if not _is_noise(n)]
        if any(_is_artifact_name(name) for name in filenames):
            return True
    return False


def _all_subdirs_are_components(subdirs: list[Path]) -> bool:
    """True if there is ≥1 sub-dir and every one is a component (not a sub-challenge)."""
    return bool(subdirs) and all(d.name.lower() in _COMPONENT_NAMES for d in subdirs)


def _iter_challenge_dirs(root: Path) -> Iterator[Path]:
    """Yield every challenge directory beneath ``root`` (see module docstring)."""
    # A ``challenge.json`` at the root itself means the whole repo is one challenge.
    if (root / "challenge.json").is_file():
        yield root
        return
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _is_noise(child.name):
            continue
        yield from _collect_challenge_dirs(child)


def _collect_challenge_dirs(d: Path) -> Iterator[Path]:
    """Recursive worker: emit the most-specific challenge-owning dirs under ``d``."""
    if _is_noise(d.name):
        return
    # Cleanest signal: a metadata file pins this exact dir as one challenge.
    if (d / "challenge.json").is_file():
        yield d
        return
    # If this dir owns artifacts directly, it *is* the challenge leaf — its sub-dirs
    # (src/, public/, solution/) are components, so we do not descend into them.
    if _has_direct_artifacts(d):
        yield d
        return
    subdirs = sorted(p for p in d.iterdir() if p.is_dir() and not _is_noise(p.name))
    # Component-only layout: every sub-dir is a challenge *part* (public/private/…),
    # so this dir is the challenge root even with nothing at its own top level
    # (justCTF/pwn.college style). Stop here rather than fragmenting into components.
    if _all_subdirs_are_components(subdirs) and _subtree_has_artifacts(d):
        yield d
        return
    # Otherwise a container (event/year/category, or the repo root) — recurse toward
    # the deeper challenge roots.
    for child in subdirs:
        yield from _collect_challenge_dirs(child)


def _load_challenge_json(d: Path) -> dict:
    """Parse ``challenge.json`` if present and valid, else an empty dict."""
    path = d / "challenge.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _year_of(segment: str) -> int | None:
    """Return a plausible 4-digit CTF year embedded in a path segment, else None."""
    for token in segment.replace("-", " ").replace("_", " ").split():
        if len(token) == 4 and token.isdigit() and 1990 <= int(token) <= 2099:
            return int(token)
    return None


def _derive_meta(
    segments: list[str], cj: dict
) -> tuple[str, str | None, str | None, int | None, str | None]:
    """Derive (title, event_name, edition, year, raw_category) from a challenge path.

    ``segments`` is the challenge dir's path relative to the repo root, split into
    parts. Best-effort and layout-agnostic: we scan the ancestor segments for a
    4-digit year, a known edition token (Quals/Finals), and a segment that maps to a
    known category; whatever ancestor segment is left over becomes the event name.
    Fields from ``challenge.json`` win when present.
    """
    title = str(cj.get("name") or (segments[-1] if segments else "unknown"))
    ancestors = segments[:-1]  # everything above the challenge dir itself

    year: int | None = None
    edition: str | None = None
    category_idx: int | None = None
    year_idx: int | None = None
    edition_idx: int | None = None

    for i, seg in enumerate(ancestors):
        if year is None and (y := _year_of(seg)) is not None:
            year, year_idx = y, i
        if seg.lower() in _EDITION_TOKENS:
            edition, edition_idx = seg, i
        if map_category(seg) is not Category.UNKNOWN:
            category_idx = i  # last-wins: the segment nearest the challenge

    raw_category = cj.get("category")
    if raw_category is None and category_idx is not None:
        raw_category = ancestors[category_idx]
    # Last resort: many flat archives encode the category as a prefix on the challenge
    # dir name (e.g. ``pwn_baby_otter`` -> pwn, ``web_mongodb`` -> web).
    if raw_category is None and "_" in title:
        prefix = title.split("_", 1)[0]
        if map_category(prefix) is not Category.UNKNOWN:
            raw_category = prefix

    # Structural wrapper segments (CTFs/, challenges/) are never the event name.
    consumed = {year_idx, edition_idx, category_idx}
    leftover = [
        seg
        for i, seg in enumerate(ancestors)
        if i not in consumed and seg.lower() not in _WRAPPER_SEGMENTS
    ]
    event_name = leftover[0] if leftover else None

    return title, event_name, edition, year, (str(raw_category) if raw_category else None)


def walk_repo(
    local_path: Path,
    seed: dict,
    repo: str,
    sha: str,
    *,
    license_info: LicenseInfo | None = None,
) -> Iterator[RawChallenge]:
    """Walk a cloned repo tree into per-challenge ``RawChallenge`` records.

    Pure and network-free: everything is derived from the on-disk tree at
    ``local_path`` plus the ``seed``/``repo``/``sha`` provenance. ``license_info`` is
    the resolved repo license; when omitted it is detected from the seed's declared
    SPDX and the repo's root LICENSE file. Factored out of :meth:`discover` so it can
    be unit-tested offline against a synthetic tree.
    """
    if license_info is None:
        license_info = detect_repo_license(local_path, declared_spdx=seed.get("license"))
    origin = _origin_for_repo(repo)
    is_official = bool(seed.get("official"))
    url = f"https://github.com/{repo}"
    retrieved_at = datetime.now(UTC)

    for chal_dir in _iter_challenge_dirs(local_path):
        rel = chal_dir.relative_to(local_path)
        rel_posix = rel.as_posix()
        segments = list(rel.parts)
        cj = _load_challenge_json(chal_dir)
        title, event_name, edition, year, raw_category = _derive_meta(segments, cj)

        # Origin-specific event naming when the path alone is uninformative.
        if event_name is None:
            event_name = (
                "Google CTF" if origin is Origin.GOOGLE_CTF else repo.split("/", 1)[-1]
            )

        source = Source(
            origin=origin,
            url=url,
            repo=repo,
            commit_sha=sha,
            path_in_repo=rel_posix or None,
            is_official=is_official,
            license=license_info,
            retrieved_at=retrieved_at,
        )
        yield RawChallenge(
            origin=origin,
            title=title,
            event_name=event_name,
            edition=edition,
            year=year,
            raw_category=raw_category,
            description=(str(cj["description"]) if cj.get("description") else None),
            flag=(str(cj["flag"]) if cj.get("flag") else None),
            local_dir=str(chal_dir),
            source=source,
            extra={
                "kind": seed.get("kind"),
                "challenge_json": cj or None,
                "path_in_repo": rel_posix,
            },
        )


class GitRepoConnector(Connector):
    """Mirror the seed GitHub repos, emitting one ``RawChallenge`` per challenge dir."""

    cli_name = "git_repo"
    origin = Origin.GITHUB

    def __init__(
        self,
        workdir: Path,
        *,
        seeds_path: Path | None = None,
        only: list[str] | None = None,
        max_challenges_per_repo: int | None = None,
    ) -> None:
        super().__init__(workdir)
        self.seeds_path = seeds_path or _DEFAULT_SEEDS
        self.only = set(only) if only else None
        self.max_challenges_per_repo = max_challenges_per_repo
        self.seeds = self._load_seeds()

    def _load_seeds(self) -> list[dict]:
        """Flatten every seed section into one list of repo entries."""
        doc = yaml.safe_load(self.seeds_path.read_text(encoding="utf-8")) or {}
        seeds: list[dict] = []
        for section in _SEED_SECTIONS:
            for entry in doc.get(section, []) or []:
                if not isinstance(entry, dict) or not entry.get("repo"):
                    continue
                if self.only is not None and entry["repo"] not in self.only:
                    continue
                seeds.append(entry)
        return seeds

    def discover(self) -> Iterator[RawChallenge]:
        for seed in self.seeds:
            repo = seed["repo"]
            local_path, sha = clone_pinned(repo, self.workdir)
            license_info = detect_repo_license(
                local_path, declared_spdx=seed.get("license")
            )
            for count, raw in enumerate(
                walk_repo(local_path, seed, repo, sha, license_info=license_info), start=1
            ):
                yield raw
                if (
                    self.max_challenges_per_repo is not None
                    and count >= self.max_challenges_per_repo
                ):
                    break


CONNECTOR = GitRepoConnector
