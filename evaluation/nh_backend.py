# uvicorn nh_backend:app --host 0.0.0.0 --port 8000
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import _bootstrap  # noqa: F401
import json
import random
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from patient_sim.memory_cards import build_cards_prompt, load_render_cards
from patient_sim.patient_prompts import build_messages, generate_patient_prompts

from repo_paths import DATA_DIR, RESULTS_DIR

random.seed(42)

app = FastAPI()
# =========================
# 1. 后端加载数据
# =========================
with open(DATA_DIR / "main_select_profiles_2.json", "r", encoding="utf-8") as f:
    ALL_PROFILES = json.load(f)

# =========================
# 2. 加载模型
# =========================
MODEL_DIR = os.getenv("DEPROFILE_MODEL_DIR", "meta-llama/Meta-Llama-3.1-8B-Instruct")


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
# [核心逻辑抽取] 模型生成函数
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
    """
    带上下文的对话（原逻辑保留给 /api/chat 使用）
    """
    messages = build_messages(system_prompt, "（患者信息已加载）", history, query)
    return _gen_reply(messages)


def get_model_reply_no_history(system_prompt: str, query: str):
    """
    无历史版本：每个问题独立，不累积上下文
    """
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
    baseline: str
    language: str
    profile_id: str

class SessionSubmitRequest(BaseModel):
    profile_id: str
    chat_history: list
    evaluation: dict | None

class BatchRunRequest(BaseModel):
    baseline: str          
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
    # print(ALL_PROFILES[profile_id])
    return {
        "profile_id": profile_id,
        "profile_info": ALL_PROFILES[profile_id]
    }


# =========================
# 5. Chat API (常规前端对话)
# =========================
@app.post("/api/chat")
def chat_api(req: ChatRequest):
    if req.profile_id not in ALL_PROFILES:
        raise HTTPException(status_code=404, detail="Profile not found.")

    profile = ALL_PROFILES[req.profile_id]

    life_event_rendering_card = build_cards_prompt(req.profile_id, profile['candidate_id'][0]['basic_id'], "life_event")
    # print(life_event_rendering_card)
    symptom_rendering_card = build_cards_prompt(req.profile_id, profile['candidate_id'][0]['basic_id'], "symptom")
    # print(symptom_rendering_card)
    prompts = generate_patient_prompts(req.profile_id, profile, req.language, life_event_rendering_card, symptom_rendering_card)
    
    if req.baseline not in prompts:
        raise HTTPException(status_code=400, detail=f"Baseline '{req.baseline}' not found.")
        
    system_prompt = prompts[req.baseline]

    # 调用公共函数
    reply = get_model_reply(system_prompt, req.history, req.query)

    new_history = req.history + [(req.query, reply)]

    return {
        "reply": reply,
        "history": new_history
    }


# =========================
# 6. Batch API (批量遍历 - 安全版)
# =========================
SAVE_DIR = str(RESULTS_DIR / "llama-3.1-8B-ablation")
BATCH_DIR = SAVE_DIR

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)

def run_batch_task(req: BatchRunRequest, timestamp: str):
    """
    后台任务：遍历所有 Profile，每跑完一个就存一个文件到专属文件夹。
    """
    total_profiles = len(ALL_PROFILES)
    print(f"[{timestamp}] 开始批量任务: Baseline={req.baseline}, Profiles={total_profiles}")

    # 1. 创建本次任务的专属文件夹
    # 文件夹名示例: batch_records/run_baseline_v1_test01_20240520_1400/
    run_dir = f"{BATCH_DIR}/run_{req.baseline}_{req.run_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    for idx, (pid, profile) in enumerate(ALL_PROFILES.items()):
        
        # 准备 Prompt
        life_event_rendering_card = build_cards_prompt(pid, profile['candidate_id'][0]['basic_id'], "life_event")
        # print(life_event_rendering_card)
        symptom_rendering_card = build_cards_prompt(pid, profile['candidate_id'][0]['basic_id'], "symptom")
        # print(symptom_rendering_card)
        prompts = generate_patient_prompts({pid: profile}, req.language, life_event_rendering_card, symptom_rendering_card)
        # print(prompts)
        if req.baseline not in prompts:
            print(f"Skipping {pid}: Baseline {req.baseline} not found.")
            continue
        system_prompt = prompts[req.baseline]
        # print(system_prompt)
        dialogue_record = []

        # 遍历问题
        for q in req.questions:
            try:
                # 无历史模式：每个问题独立生成
                reply = get_model_reply_no_history(system_prompt, q)
                dialogue_record.append({
                    "question": q,
                    "answer": reply
                })
            except Exception as e:
                print(f"Error processing {pid} on question '{q}': {e}")
                dialogue_record.append({
                    "question": q,
                    "answer": f"ERROR: {str(e)}"
                })

        # ------------------------------------------
        # 核心修改：跑完一个人，立刻存一个文件
        # ------------------------------------------
        single_result = {
            "profile_id": pid,
            "baseline_id": req.baseline,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_prompt": system_prompt,
            "dialogue": dialogue_record
        }

        file_path = f"{run_dir}/{pid}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(single_result, f, ensure_ascii=False, indent=2)

        # 打印进度
        if (idx + 1) % 10 == 0:
            print(f"[{timestamp}] Progress: {idx + 1}/{total_profiles} (Saved to {run_dir})")
        # break
    print(f"[{timestamp}] 批量任务完成。所有结果已保存在文件夹: {run_dir}")


@app.post("/api/run_batch")
def trigger_batch_run(req: BatchRunRequest, background_tasks: BackgroundTasks):
    """
    触发接口
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 放入后台运行
    background_tasks.add_task(run_batch_task, req, timestamp)

    run_dir_name = f"run_{req.baseline}_{req.run_name}_{timestamp}"
    
    return {
        "status": "started",
        "message": "Batch processing started in background.",
        "output_directory": f"{BATCH_DIR}/{run_dir_name}"
    }


# =========================
# 7. 提交 Session (前端手动评测用)
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