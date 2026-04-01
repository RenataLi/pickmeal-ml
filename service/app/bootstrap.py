from __future__ import annotations

import sys
from pathlib import Path


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parents[2], Path.cwd().resolve()]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
        # allow running from the scaffold folder itself
        if (candidate / "service").exists() and (candidate / "README.md").exists():
            return candidate

    # fallback: walk upwards from this file
    for parent in here.parents:
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    return Path.cwd().resolve()


def ensure_src_on_path() -> Path:
    project_root = get_project_root()
    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return project_root
