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
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import yaml
from loguru import logger

from ctfhoard.connectors.base import Connector
from ctfhoard.mirror import MirrorError, clone_pinned, detect_repo_license
from ctfhoard.normalize import _CATEGORY_MAP, map_category
from ctfhoard.schema import Category, LicenseInfo, Origin, RawChallenge, Source

#: Upper bound on bytes read from an untrusted ``challenge.json``. A repo can ship a
#: huge file named ``challenge.json``; a metadata probe must never slurp gigabytes, so
#: anything larger than this is treated as absent (real metadata files are tiny).
_MAX_CHALLENGE_JSON_BYTES: int = 1024 * 1024  # 1 MiB

#: Default cap (megabytes) on a *discovered* repo's size before it is cloned. Generous
#: (a legitimately large archive still fits) but finite, so an accidental multi-GB repo
#: from open-ended discovery cannot exhaust the disk. Curated seeds are never capped.
_DEFAULT_MAX_REPO_SIZE_MB: int = 2048

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
        # kCTF / Google-CTF style challenge layout: the real challenge dir wraps its
        # parts in these sub-dirs (the ``challenge`` payload, the ``healthcheck`` prober,
        # the player ``attachments``, the ``solution``, ``metadata``). Treat them as parts
        # so the PARENT is emitted as one challenge instead of each part becoming a fake
        # one. Deliberately conservative: generic words that are just as often *category
        # container* names (``web``, ``app``, ``bot``, ``chal``) are NOT listed here — a
        # ``.../web/<challenge>`` container must still recurse, not collapse.
        "challenge",
        "healthcheck",
        "health-check",
        "metadata",
        "sample",
        "samples",
        "sol",
        "sols",
        "resources",
        "static",
    }
)

#: The strongest "this dir is a challenge *part*, so my parent is the challenge" signals.
#: The mere *presence* of one of these sub-dirs pins the parent as a single challenge
#: root even when it also has oddly-named source sub-dirs (``frontend``, ``pcb``,
#: ``warrior`` …), which would otherwise fragment the challenge into its parts.
_ROOT_MARKER_NAMES = frozenset(
    {
        "attachments",
        "challenge",
        "healthcheck",
        "health-check",
        "solution",
        "solutions",
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
        # Vendored / scaffolding trees that hold *no challenge of their own*: descending
        # into them fragments a huge dependency checkout (edk2, cpython, PEGTL …) or the
        # kCTF framework into thousands of bogus, uncategorizable "challenges".
        "third_party",
        "third-party",
        "thirdparty",
        "vendor",
        "vendored",
        "kctf",
    }
)

#: Path segments that denote an edition rather than an event/category/challenge.
_EDITION_TOKENS = frozenset({"quals", "qualifiers", "finals", "final", "onsite"})

#: Whole-token category vocabulary for parsing a category out of a challenge dir NAME.
#: Derived from the normalize map so the two stay in sync, keeping only single-word keys
#: — a multi-word label like "binary exploitation" never appears as one hyphen/underscore
#: token. Values are canonical labels that :func:`map_category` resolves, so the derived
#: ``raw_category`` still normalizes correctly. A couple of common abbreviations the
#: substring-based map cannot resolve on their own are added explicitly: ``re`` (Google's
#: reverse-engineering prefix) and ``hw`` (hardware).
_NAME_CATEGORY_TOKENS: dict[str, str] = {key: key for key in _CATEGORY_MAP if " " not in key}
_NAME_CATEGORY_TOKENS.update({"re": "reverse", "hw": "hardware"})

#: Split a challenge dir name into its lower-cased tokens (hyphen/underscore/space).
_NAME_TOKEN_RE = re.compile(r"[-_\s]+")


def _category_from_name(name: str) -> str | None:
    """Derive a raw category label from a challenge directory NAME, else ``None``.

    Google-CTF and many flat archives encode the category as a token in the dir name
    (``2017-finals-crypto-bender`` -> crypto, ``pwn-hyperion`` -> pwn, ``re-corewars`` ->
    reverse, ``hw-parking`` -> hardware). We split on ``-``/``_``/space and match WHOLE
    tokens against :data:`_NAME_CATEGORY_TOKENS` — never a naive substring, so ``web``
    inside ``webhook`` cannot false-match while the short ``re``/``hw`` tokens still do.
    The first matching token wins: these layouts put the category prefix at the front.
    """
    for token in _NAME_TOKEN_RE.split(name.lower()):
        label = _NAME_CATEGORY_TOKENS.get(token)
        if label is not None:
            return label
    return None

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


def _is_challenge_root(subdirs: list[Path]) -> bool:
    """True if ``subdirs`` mark their parent as a single challenge root (not a container).

    Two signals collapse a directory to one challenge instead of fragmenting it into its
    parts:

    * every sub-dir is a component (justCTF/pwn.college ``public``+``private`` style), or
    * at least one sub-dir is a *strong* root marker (``challenge``/``attachments``/
      ``healthcheck``/``solution``). Its presence pins the parent as the challenge even
      when the parent also has oddly-named source sub-dirs (``frontend``, ``pcb`` …) that
      would otherwise be mistaken for separate challenges — the kCTF/Google-CTF layout.
    """
    return _all_subdirs_are_components(subdirs) or any(
        d.name.lower() in _ROOT_MARKER_NAMES for d in subdirs
    )


def _iter_challenge_dirs(root: Path) -> Iterator[Path]:
    """Yield every challenge directory beneath ``root`` (see module docstring).

    The repo root is special-cased to **prefer sub-challenges** over collapsing the
    whole repo into a single challenge. A multi-challenge archive — dozens/hundreds of
    ``<event>/<challenge>/…`` nested dirs (cryptohack/ctf_archive, sajjadium/ctf-archives,
    ctfs/write-ups, pwncollege/ctf-archive) — must fan out into its many nested challenge
    dirs, never swallow the entire tree as one root challenge with ``path_in_repo == '.'``.
    (Regression guarded against: a stray artifact at the archive root — e.g. cryptohack's
    top-level ``docker_deploy.py`` — otherwise made ``_has_direct_artifacts(root)`` fire
    and the root greedily emit itself as one giant challenge.)

    Order of preference at the root:

    * an explicit ``root/challenge.json`` short-circuits to exactly one challenge (the
      repo genuinely *is* one challenge);
    * otherwise recurse into the root's real (non-component) sub-dirs FIRST — if that
      yields ONE OR MORE challenges, those are the result and the root is NOT emitted;
    * ONLY when recursion yields ZERO challenges do we fall back to emitting the root
      itself as a single challenge: the repo that *is* one challenge (artifacts at the
      root, no challenge-bearing sub-dirs), a component-only root (``public``/``private``,
      ``src``/``deploy``), or a kCTF wrapper (``attachments``/``challenge``/``healthcheck``).
      In each of those the sub-dirs are the challenge's own *parts* (components), so
      recursion into them yields nothing and the fallback correctly fires.

    Nested directories keep the ordinary most-specific-leaf logic
    (:func:`_collect_challenge_dirs`), so per-repo challenge counts below the root are
    unchanged.
    """
    if _is_noise(root.name):
        return
    # Cleanest signal: an explicit challenge.json at the very root = one challenge.
    if (root / "challenge.json").is_file():
        yield root
        return
    subdirs = sorted(p for p in root.iterdir() if p.is_dir() and not _is_noise(p.name))
    # Prefer sub-challenges: recurse into the root's real sub-dirs first. Component
    # sub-dirs are parts of a *single* root challenge (not sub-challenges), so they are
    # skipped here — that is what lets a component-only / kCTF wrapper root fall through
    # to the single-challenge fallback below instead of fragmenting into its parts.
    produced = False
    for child in subdirs:
        if child.name.lower() in _COMPONENT_NAMES:
            continue
        for leaf in _collect_challenge_dirs(child):
            produced = True
            yield leaf
    if produced:
        return
    # Fallback: recursion found no nested challenge, so the root itself is a single
    # challenge when it owns artifacts directly or is a component-only / challenge-root
    # (kCTF) layout whose sub-dirs are its components.
    if _has_direct_artifacts(root) or (
        _is_challenge_root(subdirs) and _subtree_has_artifacts(root)
    ):
        yield root


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
    # Component/challenge-root layout: the sub-dirs are challenge *parts* (public/private,
    # or a kCTF challenge/attachments/healthcheck/solution wrapper), so this dir is the
    # challenge root even with nothing at its own top level. Stop here rather than
    # fragmenting into components (and never emitting a leaf titled ``challenge`` etc.).
    if _is_challenge_root(subdirs) and _subtree_has_artifacts(d):
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
        # Guard against a pathological multi-GB ``challenge.json``: a real metadata
        # file is tiny, so anything over the cap is treated as absent rather than read.
        if path.stat().st_size > _MAX_CHALLENGE_JSON_BYTES:
            return {}
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
    # Last resort: many archives encode the category in the challenge dir NAME rather than
    # as a clean ancestor segment (Google-CTF ``2017-finals-crypto-bender`` -> crypto,
    # ``pwn-hyperion`` -> pwn, ``re-corewars`` -> reverse; flat ``web_mongodb`` -> web).
    # Only consult the name when no ancestor segment supplied a category.
    if raw_category is None and segments:
        raw_category = _category_from_name(segments[-1])

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

        # Root itself is the challenge leaf (single-challenge repo): the relative path
        # is empty, so derive the title from the repo name rather than "unknown".
        if not segments:
            title = str(cj.get("name") or repo.split("/")[-1])

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
        discovered_path: Path | None = None,
        max_repo_size_mb: int = _DEFAULT_MAX_REPO_SIZE_MB,
    ) -> None:
        super().__init__(workdir)
        self.seeds_path = seeds_path or _DEFAULT_SEEDS
        self.only = set(only) if only else None
        self.max_challenges_per_repo = max_challenges_per_repo
        self.discovered_path = discovered_path
        self.max_repo_size_mb = max_repo_size_mb
        self.seeds = self._load_seeds()
        if discovered_path is not None:
            self.seeds.extend(self._load_discovered_seeds(discovered_path))

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

    def _load_discovered_seeds(self, path: Path) -> list[dict]:
        """Turn a ``discover``-produced JSONL into seed entries appended to the walk.

        Each :class:`~ctfhoard.discover.RepoCandidate` becomes a seed dict shaped like
        the YAML seeds so :func:`walk_repo` handles it identically: ``kind`` carries the
        inferred nature, ``license`` is left ``None`` (detected from the repo's own
        LICENSE at clone time — the discovery SPDX is only a hint), and ``official`` is
        ``False``. The ``only`` filter applies here too. Repos already present as
        curated seeds are skipped so they are not walked twice.
        """
        from ctfhoard.discover import load_discovered

        known = {s["repo"] for s in self.seeds}
        cap_kb = self.max_repo_size_mb * 1024
        extra: list[dict] = []
        for cand in load_discovered(path):
            repo = cand.full_name
            if repo in known:
                continue
            if self.only is not None and repo not in self.only:
                continue
            # Disk-exhaustion guard: skip open-endedly discovered repos that exceed the
            # size cap before they are ever cloned/walked. size_kb is known up front.
            if cand.size_kb > cap_kb:
                logger.warning(
                    "skipping discovered repo {} ({} KB > {} MB cap)",
                    repo,
                    cand.size_kb,
                    self.max_repo_size_mb,
                )
                continue
            known.add(repo)
            extra.append(
                {
                    "repo": repo,
                    "kind": cand.kind,
                    "license": None,
                    "official": False,
                    "note": "discovered via github search",
                }
            )
        return extra

    def discover(self) -> Iterator[RawChallenge]:
        for seed in self.seeds:
            repo = seed["repo"]
            try:
                local_path, sha = clone_pinned(repo, self.workdir)
                license_info = detect_repo_license(
                    local_path, declared_spdx=seed.get("license")
                )
            except MirrorError as exc:
                # A timed-out / failed clone must not abort the whole run: log and skip
                # this repo, continuing with the rest.
                logger.warning("skipping repo {}: {}", repo, exc)
                continue
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
