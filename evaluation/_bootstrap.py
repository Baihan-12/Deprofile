"""Ensure `src/` is on sys.path when running scripts from `evaluation/`."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_p = str(_SRC)
if _p not in sys.path:
    sys.path.insert(0, _p)
