import _bootstrap  # noqa: F401
import os
import json
import random
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from patient_sim.patient_prompts import generate_patient_prompts
from patient_sim.memory_cards import load_render_cards, build_cards_prompt

from repo_paths import API_RES_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Batch QA evaluation with OpenAI API (concurrent + tqdm + timing)")

    # OpenAI / 中转配置
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "https://api.xi-ai.cn/v1"))
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))

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
    parser.add_argument("--temperature", type=float, default=0.0)  # QA 推荐 0
    parser.add_argument("--top_p", type=float, default=1.0)        # QA 推荐 1
    parser.add_argument("--max_tokens", type=int, default=128)     # QA 推荐短一点

    # 并发
    parser.add_argument("--num_workers", type=int, default=8, help="Concurrent workers across profiles")
    parser.add_argument("--max_profiles", type=int, default=-1, help="Debug: only run first N profiles")

    # 断点续跑
    parser.add_argument("--skip_existing", action="store_true", help="Skip if pid.json already exists")

    # 重试（简单版）
    parser.add_argument("--retries", type=int, default=2, help="Retry times for API call failures")

    return parser.parse_args()


def get_openai_reply(client: OpenAI, model_name: str, system_prompt: str, history: list, query: str,
                    temperature: float, top_p: float, max_tokens: int, retries: int):
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": query})

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            # 轻量退避：0.5, 1.0, 2.0...
            time.sleep(0.5 * (2 ** attempt))

    return f"Error: {str(last_err)}"


def process_one_profile(pid: str, profile: dict, *,
                        args,
                        questions_list: list,
                        run_dir: str):
    """
    单个 profile 的完整处理（串行问答），适合在线程池里跑。
    注意：每个 worker 内部创建自己的 OpenAI client，避免线程安全/连接复用问题。
    """
    start_t = time.perf_counter()

    out_path = os.path.join(run_dir, f"{pid}.json")
    if args.skip_existing and os.path.exists(out_path):
        return pid, "skipped", 0.0, None

    # 每个 worker 建一个 client（稳）
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 构建 prompt
    try:
        basic_id = profile["candidate_id"][0]["basic_id"]
        life_event_card = build_cards_prompt(pid, basic_id, "life_event")
        symptom_card = build_cards_prompt(pid, basic_id, "symptom")

        prompts_dict = generate_patient_prompts(
            pid,
            profile,
            args.language,
            life_event_card,
            symptom_card
        )
    except Exception as e:
        elapsed = time.perf_counter() - start_t
        return pid, "failed_prompt", elapsed, str(e)

    if args.baseline not in prompts_dict:
        elapsed = time.perf_counter() - start_t
        return pid, "failed_baseline_missing", elapsed, f"Baseline '{args.baseline}' not found"

    system_prompt = prompts_dict[args.baseline]

    history = []
    dialogue_record = []

    for q in questions_list:
        reply = get_openai_reply(
            client=client,
            model_name=args.model,
            system_prompt=system_prompt,
            history=history,
            query=q,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            retries=args.retries
        )
        history.append((q, reply))
        dialogue_record.append({"question": q, "answer": reply})

    result_data = {
        "profile_id": pid,
        "baseline_id": args.baseline,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": system_prompt,
        "dialogue": dialogue_record,
        "gen_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "retries": args.retries
        }
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start_t
    return pid, "ok", elapsed, None


def run_batch_evaluation(args):
    if not args.api_key:
        raise ValueError("No API key provided. Set env OPENAI_API_KEY or pass --api_key.")

    random.seed(args.seed)

    # Load questions
    with open(args.questions_path, "r", encoding="utf-8") as f:
        questions_dict = json.load(f)
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
    save_root_dir = args.save_root_dir or str(
        API_RES_DIR / f"{args.model}_{args.run_name}_results"
    )
    os.makedirs(save_root_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{args.baseline}_{args.run_name}_{timestamp}"
    run_dir = os.path.join(save_root_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Results will be saved to: {run_dir}")
    print(f"Profiles: {len(items)} | Workers: {args.num_workers} | Model: {args.model} | Baseline: {args.baseline}")

    # Timing & stats
    global_start = time.perf_counter()
    done = 0
    ok = 0
    skipped = 0
    failed = 0
    elapsed_list = []

    # 并发执行（profile 级）
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        for pid, profile in items:
            futures.append(
                ex.submit(
                    process_one_profile,
                    pid, profile,
                    args=args,
                    questions_list=questions_list,
                    run_dir=run_dir
                )
            )

        # tqdm：按完成进度更新
        pbar = tqdm(total=len(futures), desc="Profiles", dynamic_ncols=True)
        for fut in as_completed(futures):
            pid, status, elapsed, err = fut.result()
            done += 1
            if status == "ok":
                ok += 1
                elapsed_list.append(elapsed)
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                # 失败信息别刷太多，只打印简要
                print(f"\n[WARN] {pid} -> {status}: {err}")

            # 计算 ETA（用已完成 ok 的平均耗时更稳；否则用已完成总耗时/ done）
            now = time.perf_counter()
            total_elapsed = now - global_start
            if elapsed_list:
                avg = sum(elapsed_list) / len(elapsed_list)
            else:
                avg = total_elapsed / max(done, 1)

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
    print("\n" + "=" * 40)
    print("Batch processing complete.")
    print(f"Saved in: {run_dir}")
    print(f"OK: {ok} | Skipped: {skipped} | Failed: {failed} | Total: {len(items)}")
    print(f"Total time: {total_time/60:.2f} min ({total_time:.1f} sec)")
    if elapsed_list:
        print(f"Avg per OK profile: {sum(elapsed_list)/len(elapsed_list):.2f} sec")
    print("=" * 40)


if __name__ == "__main__":
    args = parse_args()
    run_batch_evaluation(args)
