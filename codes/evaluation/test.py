import os
    # uvicorn ls_backend:app --host 0.0.0.0 --port 8000
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import generate_patient_prompts, build_messages
from datetime import datetime
from ls_utils import load_render_cards, build_cards_prompt
import os
import random
from tqdm import tqdm
# set seed
random.seed(42)

# =========================
# 1. 后端加载数据
# =========================
# 确保路径正确
with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/main_select_profiles_2.json", "r", encoding="utf-8") as f:
    ALL_PROFILES = json.load(f)

# =========================
# 2. 加载模型
# =========================
MODEL_DIR = "/hpc_stor03/sjtu_home/baihan.li/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"
# MODEL_DIR = "/hpc_stor03/sjtu_home/baihan.li/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"


# =========================
# [核心逻辑抽取] 模型生成函数
# =========================

if __name__ == "__main__":
    for profile_id in tqdm(ALL_PROFILES):
        cr_id = ALL_PROFILES[profile_id]['cr_id']
        d4_id = ALL_PROFILES[profile_id]['d4_id']

        with open(f"/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/dialogues/assessment/{profile_id}.json", "w", encoding="utf-8") as f:
            with open(f"/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/cr_dialogues/{cr_id}.json", "r", encoding="utf-8") as f1:
                cr_dialogues = json.load(f1)
            json.dump(cr_dialogues, f, ensure_ascii=False, indent=4)

        with open(f"/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/dialogues/counseling/{profile_id}.json", "w", encoding="utf-8") as f:
            with open(f"/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/d4_dialogues/{d4_id}.json", "r", encoding="utf-8") as f1:
                d4_dialogues = json.load(f1)
            json.dump(d4_dialogues, f, ensure_ascii=False, indent=4)