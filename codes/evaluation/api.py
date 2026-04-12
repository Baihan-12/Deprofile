#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch QA evaluation (Scheme A):
- For each profile, send ONLY ONE API request that answers ALL questions in a single JSON output.
- Greatly reduces repeated system prompt + history tokens and reduces API call counts.

Deps:
  pip install openai tqdm

Usage example:
  python direct_api_batch.py \
    --data_path /path/to/main_select_profiles_2.json \
    --questions_path /path/to/QA.json \
    --model gpt-4.1 \
    --base_url https://api-2.xi-ai.cn/v1 \
    --api_key YOUR_KEY \
    --baseline G1 \
    --run_name test_run \
    --language zh \
    --num_workers 16 \
    --max_tokens 2048 \
    --skip_existing
"""

import os
import json
import random
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import generate_patient_prompts
from ls_utils import build_cards_prompt
from tqdm import tqdm
from openai import OpenAI

# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Batch QA evaluation (1 request per profile)")

    # OpenAI / 中转配置
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base_url", default="https://models.sjtu.edu.cn/api/v1/")
    parser.add_argument("--api_key", default="")

    # 数据路径
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--questions_path", required=True)
    parser.add_argument("--save_root_dir", default="")

    # 实验设置
    parser.add_argument("--baseline", default="G0")
    parser.add_argument("--run_name", default="test_run")
    parser.add_argument("--language", default="zh", choices=["zh", "en"])
    parser.add_argument("--seed", type=int, default=42)

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7)  # QA 推荐 0
    parser.add_argument("--top_p", type=float, default=0.9)        # QA 推荐 1
    parser.add_argument("--max_tokens", type=int, default=1000)    # Scheme A: 一次回答很多题，建议 >= 1024

    # 并发
    parser.add_argument("--num_workers", type=int, default=8, help="Concurrent workers across profiles")
    parser.add_argument("--max_profiles", type=int, default=-1, help="Debug: only run first N profiles")

    # 断点续跑
    parser.add_argument("--skip_existing", action="store_true", help="Skip if pid.json already exists")

    # 重试
    parser.add_argument("--retries", type=int, default=2, help="Retry times for API call failures")
    parser.add_argument("--backoff_base", type=float, default=0.6, help="Backoff base seconds")

    # JSON 输出更稳：如果你的后端支持 response_format（OpenAI 新接口/部分中转支持）
    parser.add_argument("--use_response_format", action="store_true",
                        help="Use response_format={'type':'json_object'} if supported by your backend")

    return parser.parse_args()


# -------------------------
# API call (batch)
# -------------------------
def build_batch_user_prompt(questions_list: list, language: str) -> str:
    """
    Ask model to output strict JSON. We keep it simple and robust.
    """
    if language == "zh":
        prefix = (
            "你将回答以下问题。请严格输出一个 JSON 对象，格式如下：\n"
            "{\n"
            '  "answers": [\n'
            '    {"question_id": 1, "answer": "..."},\n'
            '    ...\n'
            "  ]\n"
            "}\n"
            "要求：\n"
            "1) 必须是合法 JSON（双引号、无多余文字）。\n"
            "2) 不要使用任何 Markdown 或代码块围栏（例如 ``` 或 ```json）。\n"
            "3) answers 数组长度必须等于问题数量，question_id 从 1 开始。\n"
            "4) 必须完整闭合大括号和中括号，不能截断。\n"
            "问题列表：\n"
        )
    else:
        prefix = (
            "You will answer the following questions. Output STRICTLY a JSON object in the format:\n"
            "{\n"
            '  \"answers\": [\n'
            '    {\"question_id\": 1, \"question\": \"...\", \"answer\": \"...\"},\n'
            '    ...\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "1) Must be valid JSON (double quotes, no extra text, no markdown).\n"
            "2) Do NOT use code fences or markdown (e.g., no ``` or ```json).\n"
            "3) answers length must equal number of questions; question_id starts at 1.\n"
            "4) Must close all braces/brackets completely; no truncation.\n"
            "Questions:\n"
        )

    qlines = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_list)])
    return prefix + qlines


def safe_extract_json(text: str):
    """
    Best-effort JSON extraction.
    - First try direct json.loads
    - If fails, try to find the first '{' and last '}' and parse the substring.
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    # Heuristic: extract JSON object substring
    l = text.find("{")
    r = text.rfind("}")
    if l != -1 and r != -1 and r > l:
        sub = text[l:r+1]
        try:
            return json.loads(sub)
        except Exception:
            return None
    return None


def get_openai_reply_batch(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    questions_list: list,
    *,
    language: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retries: int,
    backoff_base: float,
    use_response_format: bool,
):
    user_prompt = build_batch_user_prompt(questions_list, language)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for attempt in range(retries + 1):
        try:
            kwargs = dict(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            # Some backends support this (OpenAI Responses / newer Chat Completions style).
            if use_response_format:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            # exponential backoff with a little jitter
            sleep_s = backoff_base * (2 ** attempt) + random.random() * 0.2
            time.sleep(sleep_s)

    return f"Error: {str(last_err)}"


# -------------------------
# Per profile processing
# -------------------------
def process_one_profile(pid: str, profile: dict, *,
                        args,
                        questions_list: list,
                        questions_keys: list,
                        run_dir: str):
    """
    Process one profile:
    - build system prompt (baseline)
    - ONE batch API call to answer all questions
    - save JSON file
    """
    start_t = time.perf_counter()

    out_path = os.path.join(run_dir, f"{pid}.json")
    if args.skip_existing and os.path.exists(out_path):
        return pid, "skipped", 0.0, None

    # One client per worker for safety
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Build prompt
    try:
        basic_id = profile["candidate_id"][0]["basic_id"]
        life_event_card = build_cards_prompt(pid, basic_id, "life_event")

        prompts_dict = generate_patient_prompts(
            {pid: profile},
            args.language,
            life_event_card,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start_t
        return pid, "failed_prompt", elapsed, str(e)

    if args.baseline not in prompts_dict:
        elapsed = time.perf_counter() - start_t
        return pid, "failed_baseline_missing", elapsed, f"Baseline '{args.baseline}' not found"

    system_prompt = prompts_dict[args.baseline]

    # ---- ONE CALL for ALL QUESTIONS ----
    raw = get_openai_reply_batch(
        client=client,
        model_name=args.model,
        system_prompt=system_prompt,
        questions_list=questions_list,
        language=args.language,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        retries=args.retries,
        backoff_base=args.backoff_base,
        use_response_format=args.use_response_format,
    )

    dialogue_record = []
    parse_err = None

    parsed = safe_extract_json(raw)
    if isinstance(parsed, dict) and "answers" in parsed and isinstance(parsed["answers"], list):
        answers = parsed["answers"]
        # normalize ordering by question_id if present
        try:
            answers_sorted = sorted(
                answers,
                key=lambda x: int(x.get("question_id", 10**9))
            )
        except Exception:
            answers_sorted = answers

        for item in answers_sorted:
            # 使用 question_id (1-based) 来获取对应的 key
            qid = item.get("question_id", None)
            if qid is not None and isinstance(qid, int) and 1 <= qid <= len(questions_keys):
                key_id = questions_keys[qid - 1]  # 1-based 转 0-based
                question = questions_list[qid - 1]
            else:
                key_id = qid  # fallback
                question = ""
            
            dialogue_record.append({
                "question_id": key_id,
                "question": question,
                "answer": item.get("answer", ""),
            })
    else:
        parse_err = "Model output is not a valid JSON object with key 'answers'."

    # Fallback: store raw output (so you never lose results)
    if parse_err is not None:
        dialogue_record = [{"question_id": "BATCH_ALL", "question": "BATCH_ALL", "answer": raw}]

    result_data = {
        "profile_id": pid,
        "baseline_id": args.baseline,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": system_prompt,
        "dialogue": dialogue_record,
        "batch_mode": True,
        "parse_ok": (parse_err is None),
        "parse_error": parse_err,
        "gen_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "retries": args.retries,
            "use_response_format": args.use_response_format,
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start_t
    return pid, "ok" if parse_err is None else "ok_but_parse_failed", elapsed, parse_err


# -------------------------
# Batch runner
# -------------------------
def run_batch_evaluation(args):
    if not args.api_key:
        raise ValueError("No API key provided. Set env OPENAI_API_KEY or pass --api_key.")

    random.seed(args.seed)

    # Load questions
    with open(args.questions_path, "r", encoding="utf-8") as f:
        questions_dict = json.load(f)

    # Keep stable order: if questions_dict is an ordered dict in JSON, fine; otherwise sort by key
    # If your JSON is {"q1":"...", "q2":"..."}, sorting keeps reproducibility.
    try:
        questions_keys = sorted(questions_dict.keys())
        questions_list = [questions_dict[k] for k in questions_keys]
    except Exception:
        questions_keys = list(questions_dict.keys())
        questions_list = list(questions_dict.values())

    # Load profiles
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    with open(args.data_path, "r", encoding="utf-8") as f:
        all_profiles = json.load(f)

    items = list(all_profiles.items())
    if args.max_profiles and args.max_profiles > 0:
        items = items[:args.max_profiles]

    # Save dir
    save_root_dir = args.save_root_dir or (
        f"/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/results/"
        f"{args.model}_{args.run_name}_results"
    )
    os.makedirs(save_root_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{args.baseline}_{args.run_name}"
    run_dir = os.path.join(save_root_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Results will be saved to: {run_dir}")
    print(f"Profiles: {len(items)} | Workers: {args.num_workers} | Model: {args.model} | Baseline: {args.baseline}")
    print(f"Questions: {len(questions_list)} | One request per profile ✅")

    global_start = time.perf_counter()
    done = ok = skipped = failed = 0
    elapsed_ok = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [
            ex.submit(
                process_one_profile,
                pid, profile,
                args=args,
                questions_list=questions_list,
                questions_keys=questions_keys,
                run_dir=run_dir
            )
            for pid, profile in items
        ]

        pbar = tqdm(total=len(futures), desc="Profiles", dynamic_ncols=True)
        for fut in as_completed(futures):
            pid, status, elapsed, err = fut.result()
            done += 1

            if status in ("ok", "ok_but_parse_failed"):
                ok += 1
                elapsed_ok.append(elapsed)
                if status == "ok_but_parse_failed":
                    # Parse failed but saved raw output; print a small warning
                    print(f"\n[WARN] {pid} parse_failed: {err}")
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"\n[WARN] {pid} -> {status}: {err}")

            now = time.perf_counter()
            total_elapsed = now - global_start
            avg = (sum(elapsed_ok) / len(elapsed_ok)) if elapsed_ok else (total_elapsed / max(done, 1))
            remaining = len(futures) - done
            eta_sec = remaining * avg

            pbar.set_postfix({
                "ok": ok,
                "skip": skipped,
                "fail": failed,
                "avg_s": f"{avg:.2f}",
                "eta_min": f"{eta_sec/60:.1f}"
            })
            pbar.update(1)

        pbar.close()

    total_time = time.perf_counter() - global_start
    print("\n" + "=" * 50)
    print("Batch processing complete.")
    print(f"Saved in: {run_dir}")
    print(f"OK: {ok} | Skipped: {skipped} | Failed: {failed} | Total: {len(items)}")
    print(f"Total time: {total_time/60:.2f} min ({total_time:.1f} sec)")
    if elapsed_ok:
        print(f"Avg per OK profile: {sum(elapsed_ok)/len(elapsed_ok):.2f} sec")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    run_batch_evaluation(args)
