"""Canonical filesystem locations for the project.

Every module that needs the repo root, the model weights, or the cached initial
candidates imports from here instead of re-deriving the path from ``__file__``.
Centralizing this keeps the anchors correct no matter how deep a module sits in
the package tree.

Layout (relative to the repo root):

    <root>/config.json                    default pipeline config
    <root>/data/weights/                  NRM checkpoints
    <root>/data/initial_candidates/       cached morphology/task-pose candidates
"""

from pathlib import Path

# paths.py lives at the repo root, so PROJECT_ROOT is its own directory.
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG = PROJECT_ROOT / "config.json"

DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = DATA_DIR / "weights"
INITIAL_CANDIDATES_DIR = DATA_DIR / "initial_candidates"

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CONFIG",
    "DATA_DIR",
    "WEIGHTS_DIR",
    "INITIAL_CANDIDATES_DIR",
]
