#!/usr/bin/env python3
"""一次性将 myplan.md 中的内容整理到本仓库目录（不修改 deprofile 源目录中的文件）。"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPROFILE_ROOT = REPO.parent
ACL_EVAL = DEPROFILE_ROOT / "ACL_agent" / "evaluation"
ACL_DATA = DEPROFILE_ROOT / "ACL_agent" / "data"
CODES_SRC = DEPROFILE_ROOT / "codes"
DIALOGUES_GT = ACL_EVAL / "dialogues_gt"

SELECTED_INDICES = [
    69, 91, 99, 107, 120, 151, 559, 563, 767, 770, 911, 1008, 1100, 1136,
    1506, 1681, 1795, 1961, 2062, 2310, 2556, 2599, 2652, 2737, 2798, 2805, 2960,
]


def key_for(i: int) -> str:
    return f"{i:04d}"


def compute_stats(index_path: Path) -> dict:
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ages = [v["age"] for v in data.values()]
    genders = Counter(v["gender"] for v in data.values())
    work = Counter(v["work_status"] for v in data.values())
    marital = Counter(v["marital_status"] for v in data.values())
    dep = Counter(v["depression_risk"] for v in data.values())
    sui = Counter(v["suiside_risk"] for v in data.values())
    pos_lens = [len(v.get("positive_symptoms") or []) for v in data.values()]
    neg_lens = [len(v.get("negative_symptoms") or []) for v in data.values()]
    return {
        "num_pairs": len(data),
        "age_min": min(ages),
        "age_max": max(ages),
        "age_mean": sum(ages) / len(ages),
        "gender": dict(genders),
        "work_status": dict(work),
        "marital_status": dict(marital),
        "depression_risk": dict(sorted(dep.items())),
        "suiside_risk": dict(sorted(sui.items())),
        "positive_symptoms_per_profile_mean": sum(pos_lens) / len(pos_lens),
        "negative_symptoms_per_profile_mean": sum(neg_lens) / len(neg_lens),
    }


def write_selected_samples(index: dict, out_path: Path) -> None:
    sel = {key_for(i): index[key_for(i)] for i in SELECTED_INDICES}
    missing = [i for i in SELECTED_INDICES if key_for(i) not in index]
    if missing:
        raise SystemExit(f"Missing keys in index: {missing}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sel, f, ensure_ascii=False, indent=4)


def copy_dialogues_sample(dest_root: Path) -> None:
    for sub in ("assessment", "counseling"):
        (dest_root / sub).mkdir(parents=True, exist_ok=True)
    for i in SELECTED_INDICES:
        k = key_for(i)
        for sub in ("assessment", "counseling"):
            src = DIALOGUES_GT / sub / f"{k}.json"
            shutil.copy2(src, dest_root / sub / f"{k}.json")


def rsync_timeline(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("stmhd_symptom_timeline", "stmhd_life_event_timeline"):
        src = ACL_DATA / name
        if not src.is_dir():
            continue
        target = dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_prompts_patient(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("clinical.py", "risk.py", "personality_traits.py"):
        shutil.copy2(ACL_EVAL / "prompts" / name, dest / name)


def copy_prompts_labeling(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted(CODES_SRC.glob("*.py")):
        shutil.copy2(p, dest / p.name)


def copy_evaluation_code(dest: Path) -> None:
    """评测主代码（含 prompts 子包，供 utils 引用）。"""
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "api.py",
        "direct_api.py",
        "utils.py",
        "agent.py",
        "ls_utils.py",
        "g_eval_new.py",
        "g_cp_eval.py",
        "post_process.py",
        "run_batch.py",
        "nh_backend.py",
        "ls_backend.py",
        "cp_backend.py",
        "test_backend.py",
        "test.py",
    ):
        p = ACL_EVAL / name
        if p.is_file():
            shutil.copy2(p, dest / name)
    copy_tree(ACL_EVAL / "prompts", dest / "prompts")
    qsrc = ACL_EVAL / "questions"
    if qsrc.is_dir():
        copy_tree(qsrc, dest / "questions")
    tsrc = ACL_EVAL / "timelines"
    if tsrc.is_dir():
        copy_tree(tsrc, dest / "timelines")


def copy_pair_code(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for p in sorted(CODES_SRC.glob("*.py")):
        shutil.copy2(p, dest / p.name)


def copy_coc(dest: Path) -> None:
    src = DEPROFILE_ROOT / "ACL_agent" / "code" / "CoCAgent"
    copy_tree(src, dest)


def patch_paths_after_copy(repo: Path) -> None:
    """将硬编码绝对路径改为相对本仓库根目录的路径。"""
    eval_dir = repo / "codes" / "evaluation"
    agent = eval_dir / "agent.py"
    if agent.is_file():
        t = agent.read_text(encoding="utf-8")
        t = t.replace(
            'self.timeline_dir = os.path.join("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data", f"stmhd_{self.timeline_type}_timeline")',
            '_repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))\n'
            '        self.timeline_dir = os.path.join(_repo_root, "timeline", f"stmhd_{self.timeline_type}_timeline")',
        )
        agent.write_text(t, encoding="utf-8")

    utils = eval_dir / "utils.py"
    if utils.is_file():
        t = utils.read_text(encoding="utf-8")
        t = t.replace(
            'dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/dialogues/assessment"',
            'dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "DEPROFILE", "dialogues_sample", "assessment"))',
        )
        t = t.replace(
            'dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/dialogues/counseling"',
            'dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "DEPROFILE", "dialogues_sample", "counseling"))',
        )
        utils.write_text(t, encoding="utf-8")

    ls = eval_dir / "ls_utils.py"
    if ls.is_file():
        t = ls.read_text(encoding="utf-8")
        t = t.replace(
            'TIMELINE_MEMORY_ROOT = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/timelines"',
            'TIMELINE_MEMORY_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "timelines"))',
        )
        t = t.replace(
            'with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/main_select_profiles.json", "r") as f:',
            'with open(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "DEPROFILE", "deprofiles_complete_index.json")), "r") as f:',
        )
        ls.write_text(t, encoding="utf-8")


def sanitize_secrets(root: Path) -> None:
    """移除已复制代码中的默认 API Key（请用环境变量或命令行传入）。"""
    patterns = [
        (r'API_KEY\s*=\s*"sk-[^"]*"', 'API_KEY = os.environ.get("OPENAI_API_KEY", "")'),
        (r'default="sk-[^"]*"', 'default=""'),
    ]
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        orig = text
        for pat, repl in patterns:
            text = re.sub(pat, repl, text)
        if text != orig:
            path.write_text(text, encoding="utf-8")


def write_eval_prompt_docs(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    g_eval_src = REPO / "codes" / "evaluation" / "g_eval_new.py"
    if not g_eval_src.is_file():
        return
    text = g_eval_src.read_text(encoding="utf-8")
    m = re.search(
        r"prompt_zh\s*=\s*\"\"\"(.*?)\"\"\"",
        text,
        re.DOTALL,
    )
    body = m.group(1).strip() if m else "（见 codes/evaluation/g_eval_new.py 中 get_eval_prompt）"
    (dest / "g_eval_persona_faithfulness_zh.txt").write_text(body, encoding="utf-8")
    (dest / "README.md").write_text(
        "# 评测用 Prompt 说明\n\n"
        "- `g_eval_persona_faithfulness_zh.txt`：从 `g_eval_new.py` 抽取的人设忠实度 G-Eval 中文模板。\n"
        "- `questions/`：批量 QA 各题集 JSON（由构建脚本从 ACL_agent 复制）。\n"
        "- 批量 QA 调用约定见仓库根目录 `Readme.md` 与 `codes/evaluation/api.py` 文档字符串。\n",
        encoding="utf-8",
    )


def write_stats_json(stats: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main() -> None:
    index_path = REPO / "DEPROFILE" / "deprofiles_complete_index.json"
    stats = compute_stats(index_path)
    write_stats_json(stats, REPO / "DEPROFILE" / "dataset_statistics.json")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    write_selected_samples(index, REPO / "DEPROFILE" / "selected_samples.json")

    sample_dir = REPO / "DEPROFILE" / "dialogues_sample"
    copy_dialogues_sample(sample_dir)

    rsync_timeline(REPO / "timeline")

    (REPO / "prompts").mkdir(exist_ok=True)
    copy_prompts_labeling(REPO / "prompts" / "labeling")
    copy_prompts_patient(REPO / "prompts" / "patient")

    (REPO / "codes").mkdir(exist_ok=True)
    copy_pair_code(REPO / "codes" / "pair")
    copy_evaluation_code(REPO / "codes" / "evaluation")
    copy_coc(REPO / "codes" / "CoCAgent")
    patch_paths_after_copy(REPO)
    sanitize_secrets(REPO / "codes" / "evaluation")

    qdst = REPO / "prompts" / "evaluation"
    if (REPO / "codes" / "evaluation" / "questions").is_dir():
        if (qdst / "questions").exists():
            shutil.rmtree(qdst / "questions")
        shutil.copytree(
            REPO / "codes" / "evaluation" / "questions",
            qdst / "questions",
        )
    write_eval_prompt_docs(qdst)

    # patient 模块内时间线路径改为相对仓库（便于阅读；运行 codes 时仍可通过环境变量覆盖）
    clin = REPO / "prompts" / "patient" / "clinical.py"
    if clin.is_file():
        c = clin.read_text(encoding="utf-8")
        c = c.replace(
            'SYMPTOM_TIMELINE_DIR = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/stmhd_symptom_timeline"',
            "SYMPTOM_TIMELINE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'timeline', 'stmhd_symptom_timeline'))",
        )
        clin.write_text(c, encoding="utf-8")

    print("OK:", REPO)
    print("  pairs:", stats["num_pairs"], "selected_samples:", len(SELECTED_INDICES))
    print("  timeline ->", REPO / "timeline")
    print("  dialogues_sample ->", sample_dir)


if __name__ == "__main__":
    main()
