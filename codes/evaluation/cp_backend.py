import os
# uvicorn cp_backend:app --host 0.0.0.0 --port 8000
    # python3 run_batch.py -i EEYORE -u http://localhost:8000/api/run_batch 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import random
import time

random.seed(42)

app = FastAPI()

# =========================
# 1. 加载数据（Patient-Psi-CM）
# =========================
PROFILE_PATH = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/eeyore_profile.json"

with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    _profile_list = json.load(f)
ALL_PROFILES = {p["id_source"]: p for p in _profile_list}



# PROFILE_PATH = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/Patient-Psi-CM_dataset.json"
# with open(PROFILE_PATH, "r", encoding="utf-8") as f:
#     _profile_list = json.load(f)
# ALL_PROFILES = {p["id"]: p for p in _profile_list}

# =========================
# 2. 加载模型
# =========================
MODEL_DIR = "/hpc_stor03/sjtu_home/baihan.li/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"
# MODEL_DIR = "/hpc_stor03/sjtu_home/baihan.li/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Loading model from {MODEL_DIR} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map="cuda:0",
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully.")


# =========================
# System prompt 构造（与 api_comparison 一致）
# =========================
def profile_to_system_prompt(profile: dict) -> str:
    # 直接把档案 JSON 放入，并加中文指令
    body = json.dumps(profile, ensure_ascii=False, indent=2)
    return f"请始终用中文回答问题。以下是病人档案，你要扮演这个病人，回答所有问题：{body}"


# =========================
# 生成函数
# =========================
def _gen_reply(messages: list) -> str:
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
    return reply


def get_model_reply(system_prompt: str, history: list, query: str):
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})
    return _gen_reply(messages)


def get_model_reply_no_history(system_prompt: str, query: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return _gen_reply(messages)


# =========================
# 3. 请求数据结构
# =========================
class ChatRequest(BaseModel):
    history: list
    query: str
    baseline: str = "CMP"
    language: str = "zh"
    profile_id: str


class SessionSubmitRequest(BaseModel):
    profile_id: str
    chat_history: list
    evaluation: dict | None


class BatchRunRequest(BaseModel):
    baseline: str = "CMP"
    questions: list[str]
    language: str = "zh"
    run_name: str = "default"


# =========================
# 4. Profile API
# =========================
@app.get("/api/get_profile")
def get_profile(profile_id: str):
    if profile_id not in ALL_PROFILES:
        raise HTTPException(status_code=404, detail=f"Profile ID '{profile_id}' not found.")
    return {
        "profile_id": profile_id,
        "profile_info": ALL_PROFILES[profile_id]
    }


# =========================
# 5. Chat API
# =========================
@app.post("/api/chat")
def chat_api(req: ChatRequest):
    if req.profile_id not in ALL_PROFILES:
        raise HTTPException(status_code=404, detail="Profile not found.")

    profile = ALL_PROFILES[req.profile_id]
    system_prompt = profile_to_system_prompt(profile)

    reply = get_model_reply(system_prompt, req.history, req.query)
    new_history = req.history + [(req.query, reply)]

    return {
        "reply": reply,
        "history": new_history
    }


# =========================
# 6. Batch API（无历史，逐问独立）
# =========================
SAVE_DIR = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/results/qwen3-4B-ablation"
BATCH_DIR = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/results/qwen3-4B-ablation"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)


def run_batch_task(req: BatchRunRequest, timestamp: str):
    run_dir = f"{BATCH_DIR}/run_{req.baseline}_{req.run_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    total_profiles = len(ALL_PROFILES)
    print(f"[{timestamp}] 开始批量任务: Baseline={req.baseline}, Profiles={total_profiles}")

    for idx, (pid, profile) in enumerate(ALL_PROFILES.items()):
        system_prompt = profile_to_system_prompt(profile)
        dialogue_record = []

        for q in req.questions:
            try:
                reply = get_model_reply_no_history(system_prompt, q)
                dialogue_record.append({"question": q, "answer": reply})
            except Exception as e:
                dialogue_record.append({"question": q, "answer": f"ERROR: {str(e)}"})

        result_data = {
            "profile_id": pid,
            "baseline_id": req.baseline,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": system_prompt,
            "dialogue": dialogue_record
        }

        file_path = f"{run_dir}/{pid}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        if (idx + 1) % 10 == 0:
            print(f"[{timestamp}] Progress: {idx + 1}/{total_profiles} (Saved to {run_dir})")

    print(f"[{timestamp}] 批量任务完成。所有结果已保存在文件夹: {run_dir}")


@app.post("/api/run_batch")
def trigger_batch_run(req: BatchRunRequest, background_tasks: BackgroundTasks):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    background_tasks.add_task(run_batch_task, req, timestamp)
    run_dir_name = f"run_{req.baseline}_{req.run_name}_{timestamp}"
    return {
        "status": "started",
        "message": "Batch processing started in background.",
        "output_directory": f"{BATCH_DIR}/{run_dir_name}"
    }


# =========================
# 7. Session 提交
# =========================
@app.post("/api/submit_session")
def submit_session(req: SessionSubmitRequest):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{SAVE_DIR}/session_{req.profile_id}_{timestamp}.json"

    data = {
        "profile_id": req.profile_id,
        "chat_history": req.chat_history,
        "evaluation": req.evaluation,
        "timestamp": timestamp
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "ok", "file": filepath}