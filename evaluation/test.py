"""One-off: copy raw CR/D4 dialogues into `data/dialogues/{assessment,counseling}/` by profile id."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import json
import random

from tqdm import tqdm

from repo_paths import DATA_DIR

random.seed(42)

with open(DATA_DIR / "main_select_profiles_2.json", "r", encoding="utf-8") as f:
    ALL_PROFILES = json.load(f)

_assessment_dir = DATA_DIR / "dialogues" / "assessment"
_counseling_dir = DATA_DIR / "dialogues" / "counseling"
_assessment_dir.mkdir(parents=True, exist_ok=True)
_counseling_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for profile_id in tqdm(ALL_PROFILES):
        cr_id = ALL_PROFILES[profile_id]["cr_id"]
        d4_id = ALL_PROFILES[profile_id]["d4_id"]

        with open(_assessment_dir / f"{profile_id}.json", "w", encoding="utf-8") as f:
            with open(DATA_DIR / "cr_dialogues" / f"{cr_id}.json", "r", encoding="utf-8") as f1:
                cr_dialogues = json.load(f1)
            json.dump(cr_dialogues, f, ensure_ascii=False, indent=4)

        with open(_counseling_dir / f"{profile_id}.json", "w", encoding="utf-8") as f:
            with open(DATA_DIR / "d4_dialogues" / f"{d4_id}.json", "r", encoding="utf-8") as f1:
                d4_dialogues = json.load(f1)
            json.dump(d4_dialogues, f, ensure_ascii=False, indent=4)
