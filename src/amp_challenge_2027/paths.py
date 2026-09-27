"""Path resolution shared across the package.

The organizers' validator runs the entry point with the repository root as the working
directory, but developers run scripts from anywhere. Relative data and checkpoint paths
must resolve in both cases, so every module that opens a bundled file goes through here.
"""

from __future__ import annotations

from pathlib import Path

# .../src/amp_challenge_2027/paths.py -> repository root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve ``path`` against the current directory, falling back to the repo root.

    Raises ``FileNotFoundError`` naming both locations tried, so a misconfigured run
    fails with a message that says where to look rather than a bare missing-file error.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate
    fallback = REPO_ROOT / path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"{path!r} not found -- looked in {Path.cwd()} and {REPO_ROOT}"
    )
