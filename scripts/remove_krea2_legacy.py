from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MMDIT = ROOT / "extensions_built_in/diffusion_models/krea2/src/mmdit.py"
IMMUTABLE = ROOT / "extensions_built_in/diffusion_models/krea2/src/immutable_arena.py"
KREA2 = ROOT / "extensions_built_in/diffusion_models/krea2/krea2.py"
TRAIN_PROCESS = ROOT / "jobs/process/BaseSDTrainProcess.py"


class CleanupError(RuntimeError):
    pass


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, source: str) -> None:
    ast.parse(source, filename=str(path))
    path.write_text(source, encoding="utf-8", newline="\n")


def _class