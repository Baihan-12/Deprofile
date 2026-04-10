"""Repository root and common paths (no hard-coded cluster home directories)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
EVAL_DIR = REPO_ROOT / "evaluation"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = EVAL_DIR / "results"
CLEANED_RESULT_DIR = EVAL_DIR / "cleaned_result"
API_RES_DIR = EVAL_DIR / "api_res"
