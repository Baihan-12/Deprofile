# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import os
import random

from repo_paths import DATA_DIR
from utils import generate_patient_prompts, build_messages

app = FastAPI()

# =========================
# 1. 后端加载数据（profile + timeline）
# =========================

with open(DATA_DIR / "deprofiles_main_index.json", "r", encoding="utf-8") as f:
    ALL_PROFILES = json.load(f)



# =========================
# 2. 加载模型
# =========================
MODEL_DIR = os.getenv("DEPROFILE_MODEL_DIR", "Qwen/Qwen3-8B")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)
model.eval()


# =========================
# 3. 请求数据结构
# =========================
class ChatRequest(BaseModel):
    history: list
    query: str
    baseline: str
    language: str
    profile_id: str


class SessionSubmitRequest(BaseModel):
    user_name: str
    profile_id: str
    chat_history: list
    evaluation: dict | None


# -------------------------------------------
# 4. Profile API
# -------------------------------------------

@app.get("/api/random_profile")
def random_profile():
    pid = random.choice(list(ALL_PROFILES.keys()))
    return {
        "profile_id": pid,
        "profile_info": ALL_PROFILES[pid]
    }

# =========================
# 5. Chat API
# =========================
@app.post("/api/chat")
def chat_api(req: ChatRequest):

    profile = ALL_PROFILES[req.profile_id]
    prompts = generate_patient_prompts(
        profile,
        req.language
    )
    system_prompt = prompts[req.baseline]

    messages = build_messages(system_prompt, "（患者信息已加载）", req.history, req.query)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()

    new_history = req.history + [(req.query, reply)]

    return {
        "reply": reply,
        "history": new_history
    }


# =========================
# 5. 保存对话与评测
# =========================
SAVE_DIR = "records"
os.makedirs(SAVE_DIR, exist_ok=True)


@app.post("/api/submit_session")
def submit_session(req: SessionSubmitRequest):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{SAVE_DIR}/session_{timestamp}.json"

    data = {
        "user_name": req.user_name,
        "profile_id": req.profile_id,
        "chat_history": req.chat_history,
        "evaluation": req.evaluation
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "file": filepath}