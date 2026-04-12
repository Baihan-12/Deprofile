import json
import os
import pandas as pd
import numpy as np  # 【新增】用于计算标准差
from tqdm import tqdm
from openai import OpenAI
from utils import evaluate_patient_prompts
from ls_utils import build_cards_prompt

# ================= 配置区域 =================
# SJTU-API
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.xi-ai.cn/v1"
# MODEL_NAME = "gpt-4o"

# API_KEY = "ms-2afdfd96-49af-46b5-b1c1-50dccac6a957"
# BASE_URL = "https://api-inference.modelscope.cn/v1/"
# # 【修改点1】这里不再是一个单独的文件路径，而是一个字典
# # 格式： "Baseline名称": "对应的清洗后数据路径"
MODEL_NAME = "deepseek-v3.2-exp"
MY_MODEL_NAME = "gpt-5-mini"

BASELINES_LIST = ["G6", "G7", "G5", "G2", "G1", "G0"]

BASELINES_MAP = {
    baseline: f"/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/cleaned_result/{MODEL_NAME}/{baseline}"
    for baseline in BASELINES_LIST
}

PROFILE_FILE = "/Users/libaihan/Documents/Research/evaluation/acl_-agent_-evaluation-master/data/main_select_profiles_2.json"
NUM_WORKERS = 4  # 并行线程数，可按需调整


SCHEMA = {
  "realism": None,
  "persona_faithfulness": None,
  "event_richness": None,
  "symptom_consistency": None,
  "overall": None,
  "reasoning": {
    "realism": "",
    "persona_faithfulness": "",
    "event_richness": "",
    "symptom_consistency": "",
    "overall": ""
  },
  "extracted_events": [
    {
      "category": "",
      "time_bucket": "",
      "time_expression": "",
      "text": ""
    }
  ]
}
 
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def format_profile_to_text(profile_data):
    """
    【核心工具】将结构化的 Profile 字典转换为 GPT-4 可读的自然语言描述
    """
    if not isinstance(profile_data, dict):
        return str(profile_data)

    # 1. 基础信息
    age = profile_data.get('age', 'Unknown')
    gender = profile_data.get('gender', 'Unknown')
    work_status = profile_data.get('work_status', 'Unknown')
    marital_status = profile_data.get('marital_status', 'Unknown')
    # 简单的映射，比如 F -> Female
    gender_str = "Female" if gender == "F" else "Male" if gender == "M" else str(gender)
    
    # 2. 提取大五人格 (可选：你可以把数字转为 High/Low 描述，这里直接给数字 GPT-4 也能懂)
    big5 = profile_data.get('big_five', {})
    big5_str = ", ".join([f"{k}: {v}" for k, v in big5.items()])

    # 3. 提取症状 (非常重要)
    # 去掉 "任务-" 这种前缀，只保留核心描述，让 prompt 更干净
    positive_symptoms = profile_data.get('positive_symptoms', [])
    negative_symptoms = profile_data.get('negative_symptoms', [])
    clean_positive_symptoms = [s.replace("任务-", "") for s in positive_symptoms]
    clean_negative_symptoms = [s.replace("任务-", "") for s in negative_symptoms]
    symptoms_str = "阳性症状："+ "; ".join(clean_positive_symptoms) + ";\n " + "阴性症状:" + "; ".join(clean_negative_symptoms)

    # 4. 提取总结
    summation = profile_data.get('summation', '')

    # 5. 组装最终文本
    formatted_text = f"""
    - **Demographics:** Age {age}, {gender_str}.
    - **Personality (Big Five):** {big5_str}
    - **Clinical Diagnosis/Summary:** {summation}
    - **Specific Symptoms:** {symptoms_str}
    """
    return formatted_text.strip()


def get_eval_prompt(dialogue, patient_prompt_text, model_name):
    """
    构造 G-Eval Prompt：把病人模型看到的prompt也给评估模型
    """
    # 避免太长：截断一下（你按需要调）
    MAX_CHARS = 12000

    prompt_zh = f"""你是一位临床心理学专家，也是角色扮演对话系统的评估专家。你的任务是基于【病人档案】与【病人模型生成回答时实际看到的Prompt】来评估该对话中病人回答的质量。


【1. 病人模型在生成回答时实际看到的 System Prompt（重要）, 包含了病人档案与约束条件】
{patient_prompt_text}

【3. 对话上下文（多轮，question=医生，answer=病人）】
{dialogue}

【4. 评分维度（每项 1-5 整数）】
A) realism（真实度）：病人回答像不像真人在真实交流（自然表达、细节、犹豫、连贯性），避免AI模板腔。
B) persona_faithfulness（人设忠实度）：是否符合人口学信息与大五人格特质（不矛盾、表达风格一致）。
C) event_richness（事件丰富度与时间多样性）：对话中明确提到的生活事件是否更具体、类别更丰富、时间跨度更多样， 提到具体时间点的对话要加分。
D) symptom_consistency（症状一致性）：对话表现的症状与Profile中的阳性/阴性症状是否一致，是否前后矛盾或乱加关键症状。

【5. 输出要求】
- 严格输出 JSON，不要 markdown，不要多余文本。
- 给出每项分数、简短理由与证据要点。
- 同时提取对话中“明确提到的生活事件”列表 extracted_events（如果没有就输出空数组）。

输出 JSON schema 如下：
{json.dumps(SCHEMA, ensure_ascii=False, indent=2)}

"""
    return prompt_zh




def run_evaluation(baselines_dict, profile_path, output_summary_file="geval_summary.csv"):
    """
    参数:
    - baselines_dict: { "ModelName": "FilePath" }
    - profile_path: profile.json 的路径
    - output_summary_file: 最终汇总结果的保存路径
    """
    
    # 创建输出目录
    details_dir = os.path.join(os.path.dirname(output_summary_file), f"{MODEL_NAME}_g_eval_details")
    os.makedirs(details_dir, exist_ok=True)


    # 1. 加载 Profile 字典 (只加载一次)
    print(f"正在加载 Profile 数据: {profile_path}")
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile_map = json.load(f)

    # 加载 baseline 的 system prompt
    baseline_prompts_dict = {}
    for pid, profile in profile_map.items():
        basic_id = profile["candidate_id"][0]["basic_id"]
        life_event_card = build_cards_prompt(pid, basic_id, "life_event")

        baseline_prompts_dict[pid] = evaluate_patient_prompts(
            {pid: profile},
            "zh",
            life_event_card,
        )

    # 用于存放最终的统计结果 (Mean, Std)
    final_summary_list = []

    # 【修改点2】外层循环：遍历每一个 Baseline
    from concurrent.futures import ThreadPoolExecutor, as_completed

    for model_name, file_path in baselines_dict.items():
        print(f"\n" + "="*40)
        print(f"🚀 正在评估模型: {model_name}")
        print(f"📂 文件路径: {file_path}")
        print("="*40)
        
        # 读取该 Baseline 的数据
        try:
            data = {}
            for pfile in os.listdir(file_path):
                with open(os.path.join(file_path, pfile), 'r', encoding='utf-8') as f:
                    data[pfile.split('.')[0]] = json.load(f)
        except Exception as e:
            print(f"❌ 无法读取文件 {file_path}, 跳过。错误: {e}")
            continue



        # 存储当前 Baseline 的所有分数
        # 存储当前 Baseline 的所有分数（按维度）
        dim_scores = {
            "realism": [],
            "persona_faithfulness": [],
            "event_richness": [],
            "symptom_consistency": [],
            "overall": []
        }
        current_model_details = []


        def eval_one(pid, item):
            try:
                _pid = pid
                if _pid not in profile_map and str(int(_pid)) in profile_map:
                    _pid = str(int(_pid))
                if _pid not in profile_map:
                    return None

                dialogue = item.get('dialogue', [])
                prompt = get_eval_prompt(dialogue, baseline_prompts_dict[_pid][model_name], model_name)

                local_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
                response = local_client.chat.completions.create(
                    model=MY_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant evaluating dialogue consistency."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                if "<think>" in content:
                    import re
                    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                if "```json" in content:
                    content = content.replace("```json", "").replace("```", "").strip()

                eval_res = json.loads(content)

                realism = int(eval_res.get("realism", 0) or 0)
                persona = int(eval_res.get("persona_faithfulness", 0) or 0)
                event_rich = int(eval_res.get("event_richness", 0) or 0)
                symptom = int(eval_res.get("symptom_consistency", 0) or 0)

                overall = eval_res.get("overall", None)
                if overall is None or int(overall or 0) == 0:
                    vals = [realism, persona, event_rich, symptom]
                    vals = [v for v in vals if v > 0]
                    overall = int(round(sum(vals) / len(vals))) if vals else 0
                else:
                    overall = int(overall)

                detail = {
                    "profile_id": _pid,
                    "realism": realism,
                    "persona_faithfulness": persona,
                    "event_richness": event_rich,
                    "symptom_consistency": symptom,
                    "overall": overall,
                    "reasoning": eval_res.get("reasoning", {}),
                    "extracted_events": eval_res.get("extracted_events", []),
                }
                return _pid, realism, persona, event_rich, symptom, overall, detail
            except Exception as e:
                print(f"Error: {e}")
                return None

        futures = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            for pid, item in data.items():
                futures.append(ex.submit(eval_one, pid, item))

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {model_name}"):
                res = fut.result()
                if not res:
                    continue
                pid, realism, persona, event_rich, symptom, overall, detail = res
                dim_scores["realism"].append(realism)
                dim_scores["persona_faithfulness"].append(persona)
                dim_scores["event_richness"].append(event_rich)
                dim_scores["symptom_consistency"].append(symptom)
                dim_scores["overall"].append(overall)
                current_model_details.append(detail)

        def mean_std(arr):
            if not arr:
                return (np.nan, np.nan)
            return (float(np.mean(arr)), float(np.std(arr)))

        if dim_scores["overall"]:
            m_r, s_r = mean_std(dim_scores["realism"])
            m_p, s_p = mean_std(dim_scores["persona_faithfulness"])
            m_e, s_e = mean_std(dim_scores["event_richness"])
            m_s, s_s = mean_std(dim_scores["symptom_consistency"])
            m_o, s_o = mean_std(dim_scores["overall"])

            print(
                f"✅ {model_name} 完成。\n"
                f"  realism:               {m_r:.4f} (Std: {s_r:.4f})\n"
                f"  persona_faithfulness:  {m_p:.4f} (Std: {s_p:.4f})\n"
                f"  event_richness:        {m_e:.4f} (Std: {s_e:.4f})\n"
                f"  symptom_consistency:   {m_s:.4f} (Std: {s_s:.4f})\n"
                f"  overall:               {m_o:.4f} (Std: {s_o:.4f})"
            )


            # 保存详细结果
            with open(os.path.join(details_dir, f"{model_name}.json"), "w", encoding="utf-8") as f:
                json.dump(current_model_details, f, ensure_ascii=False, indent=4)

            final_summary_list.append({
                "Model_Name": model_name,

                "Realism_Mean": m_r,
                "Realism_Std": s_r,

                "Persona_Mean": m_p,
                "Persona_Std": s_p,

                "EventRich_Mean": m_e,
                "EventRich_Std": s_e,

                "Symptom_Mean": m_s,
                "Symptom_Std": s_s,

                "Overall_Mean": m_o,
                "Overall_Std": s_o,

                "Sample_Size": len(dim_scores["overall"])
            })
        else:
            print(f"⚠️ {model_name} 没有产生有效分数。")


    # 4. 最后输出总表
    if final_summary_list:
        df_summary = pd.DataFrame(final_summary_list)
        # 按 Overall_Mean 排序（原 G_Eval_Mean 不存在会报 KeyError）
        if "Overall_Mean" in df_summary.columns:
            df_summary = df_summary.sort_values(by="Overall_Mean", ascending=False)
        df_summary.to_csv(output_summary_file, index=False, encoding='utf-8-sig')
        print(f"\n🏆 所有模型评估完成！汇总结果已保存至: {output_summary_file}")
        print(df_summary)
    else:
        print("未生成任何结果。")

if __name__ == "__main__":
    # 调用函数，传入配置好的字典
    run_evaluation(BASELINES_MAP, PROFILE_FILE, f"../geval_res/{MODEL_NAME}_g_eval_summary.csv")