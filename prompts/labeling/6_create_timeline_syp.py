import pandas as pd
import json
import os
from tqdm import tqdm

LIFE_EVENT_COLUMNS = [
    'Career', 'Death', 'Education', 'Financial', 'Health',
    'Identity', 'Legal', 'Lifestyle_Change', 'New_Birth_in_Family',
    'Relationships_Changes', 'Relocation', 'Societal'
]

ID_JSON_PATH = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/stmhd_life_event_timeline/"



import os
import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from create_timeline_le_5 import clean_tweet

def load_valid_ids(json_path):
    """Load the user IDs that should be kept."""
    valid_ids = os.listdir(json_path)
    valid_ids = [id.split(".")[0] for id in valid_ids]
    return valid_ids


def filter_and_clean_csv(input_csv_path, chunk_size=50000):
    """
    Process a large CSV by:
    1. Dropping life event columns
    2. Keeping only rows where user_id is in id.json
    3. Saving output by overwriting the original file
    """

    valid_ids = load_valid_ids(ID_JSON_PATH)
    output_path =  "../data/deprofile_symptoms.csv" # 覆盖原文件

    # 需要先写一个空文件，等下 append
    if os.path.exists(output_path):
        os.remove(output_path)

    # 为了保留表头，第一次写入时需要特殊处理
    first_chunk = True

    # 分块读取
    for chunk in tqdm(
        pd.read_csv(input_csv_path, chunksize=chunk_size),
        desc="Processing CSV chunks",
        unit="chunk"
    ):
        # 仅保留 user_id 在 valid_ids 中的行
        chunk = chunk[chunk["user_id"].astype(str).isin(valid_ids)]

        # 删除生活事件列（如果某些列不存在也不会报错）
        chunk = chunk.drop(columns=[col for col in LIFE_EVENT_COLUMNS if col in chunk.columns])

        # 写入 CSV（分块 append）
        chunk.to_csv(
            output_path,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,  # 第一次写入带表头
            encoding="utf-8"
        )

        first_chunk = False


def main2():
    input_csv_path = "../data/tweetmind_verification_results_normalized.csv"  # TODO: 在这里改路径

    print("开始处理大规模 symptom timeline CSV...")
    filter_and_clean_csv(input_csv_path)
    print("处理完成！结果已覆盖原文件。")



# 这就是你症状相关的所有列（根据你给我的表头整理的）
SYMPTOM_COLUMNS = [
    "Anxious_Mood", "Autonomic_symptoms", "Cardiovascular_symptoms",
    "Catatonic_behavior", "Decreased_energy_tiredness_fatigue",
    "Depressed_Mood", "Gastrointestinal_symptoms", "Genitourinary_symptoms",
    "Hyperactivity_agitation", "Impulsivity", "Inattention", "Indecisiveness",
    "Respiratory_symptoms", "Suicidal_ideas", "Worthlessness_and_guilty",
    "avoidance_of_stimuli", "compensatory_behaviors_to_prevent_weight_gain",
    "compulsions", "diminished_emotional_expression",
    "do_things_easily_get_painful_consequences",
    "drastical_shift_in_mood_and_energy", "fear_about_social_situations",
    "fear_of_gaining_weight", "fears_of_being_negatively_evaluated",
    "flight_of_ideas", "intrusion_symptoms", "loss_of_interest_or_motivation",
    "more_talktive", "obsession", "panic_fear", "pessimism", "poor_memory",
    "sleep_disturbance", "somatic_muscle", "somatic_symptoms_others",
    "somatic_symptoms_sensory", "weight_and_appetite_change",
    "Anger_Irritability"
]


import pandas as pd
import os
import json
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
"""
contains 清理单个prompt的函数
"""

import re

import pandas as pd
import os


def clean_model_output_column(df):
    """
    清洗 df['model_output']：
    - 只取第一个 token
    - 去掉结尾的小数点
    - 尝试转成 int
    - 解析失败的行丢弃
    """
    def _parse_one(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if not s:
            return None

        # 只取第一个“空格分开的部分”
        first = s.split()[0]   # 例如 '1.' / '0.' / '1'
        first = first.strip()

        # 去掉结尾的小数点
        if first.endswith('.'):
            first = first[:-1]

        try:
            return int(first)
        except ValueError:
            return None

    tqdm.pandas(desc="Parsing model_output")
    df["model_output_clean"] = df["model_output"].progress_apply(_parse_one)

    bad = df["model_output_clean"].isna().sum()
    print(f"⚠️ 无法解析的 model_output 行数: {bad}")

    # 丢掉解析不了的行
    df = df[df["model_output_clean"].notna()].copy()
    df["model_output"] = df["model_output_clean"].astype(int)
    df = df.drop(columns=["model_output_clean"])

    return df

def extract_life_event_groups(input_csv, output_dir):
    """
    按 life_event 分类，分别输出 model_output=1 和 model_output=0 的数据。
    并在命令行打印每类数量。
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 建议全部按字符串读取，避免 dtype 问题
    df = pd.read_csv(input_csv, dtype=str)

    # model_output 转成 int（如果是字符串的话）
    df = clean_model_output_column(df)

    # 获取所有 life_event 类型
    life_events = df["life_event"].unique()

    print("共检测到 life_event 类型数量：", len(life_events))
    print("-" * 50)

    for le in tqdm(life_events):
        subset = df[df["life_event"] == le]

        pos = subset[subset["model_output"] == 1]
        neg = subset[subset["model_output"] == 0]

        # 输出 CSV 文件
        le_clean = le.replace("/", "_").replace(" ", "_")  # 防止非法文件名

        pos_path = os.path.join(output_dir, f"{le_clean}_now_and_positive.csv")
        neg_path = os.path.join(output_dir, f"{le_clean}_notnow_but_positive.csv")

        pos.to_csv(pos_path, index=False)
        neg.to_csv(neg_path, index=False)

        # 在终端打印数量
        print(f"Life Event: {le}")
        print(f"  Positive(model_output=1): {len(pos)} 条")
        print(f"  Negative(model_output=0): {len(neg)} 条")
        print("-" * 50)





def clean_tweet(text):
    """
    删除 tweet 文本中的所有 http / https 链接
    """
    if not isinstance(text, str):
        return text

    # 清理所有 http / https 开头的 URL（直到空格或句子结束）
    cleaned = re.sub(r'http\S+', '', text)

    # 再清理因为删除 URL 留下的多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned




def load_reference_dates(ref_csv_path, use_reference="label"):
    """
    读取 user 的 reference date（例如 earliest_date 或 latest_date）
    返回 dict: {user_id: reference_datetime}
    """
    ref_df = pd.read_csv(ref_csv_path)
    ref_map = {}
    for _, row in ref_df.iterrows():
        user = str(row["user_id"])
        ref_date = row[use_reference]
        ref_map[user] = datetime.fromisoformat(ref_date)
    return ref_map


def convert_to_relative_timestamp(tweet_date, reference_date):
    """
    将真实时间转换为相对天数（int），只按日期计算，自动避开时区问题
    """
    # 转成 date（丢弃时区和具体时分秒）
    tweet_d = tweet_date.date()
    ref_d = reference_date.date()
    return (tweet_d - ref_d).days

def build_symptom_relative_timeline(input_csv, ref_map, output_dir,
                                    raw_dir="/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered",
                                    time_col="date"):
    """
    根据 symptom-only CSV + reference dates 生成「相对时间的 symptom timeline」。
    逻辑与上午的 build_relative_timeline 完全一致，只是：
    - 需要根据 user_id / note_id 从原始 CSV 找真实发帖日期
    - timeline 项从 life_event 换成 symptom
    - 每个用户额外添加 symptoms 列表
    """

    df = pd.read_csv(input_csv)
    df["user_id"] = df["user_id"].astype(str)
    df["note_id"] = df["note_id"].astype(str)

    # 保留有任意一个 symptom == 1 的行（减少计算量）
    def has_any_symptom(row):
        for col in SYMPTOM_COLUMNS:
            if col in row and pd.notna(row[col]):
                try:
                    if float(row[col]) == 1.0:
                        return True
                except:
                    continue
        return False

    df = df[df.apply(has_any_symptom, axis=1)]

    # 按你的习惯：用 defaultdict(list)
    timelines = defaultdict(list)

    # cache 避免重复加载每个用户的 raw CSV
    raw_cache = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing symptom rows"):
        user = str(row["user_id"])
        note_id = str(row["note_id"])

        if user not in ref_map:
            continue

        # 先找用户的原始 CSV（含真实 date/timestamp）
        if user not in raw_cache:
            raw_path = os.path.join(raw_dir, f"{user}.csv")
            if os.path.exists(raw_path):
                raw_df = pd.read_csv(raw_path, dtype={"tweet_id": str})
                if time_col in raw_df.columns:
                    raw_df[time_col] = pd.to_datetime(raw_df[time_col], errors="coerce")
                else:
                    raw_df = None
            else:
                raw_df = None
            raw_cache[user] = raw_df

        raw_df = raw_cache[user]
        if raw_df is None:
            continue

        # 根据 note_id 找真实发帖时间
        hit = raw_df[raw_df["tweet_id"] == note_id]
        if hit.empty:
            continue

        real_date = hit[time_col].iloc[0]
        if pd.isna(real_date):  
            continue

        # 计算相对时间（沿用你上午同款函数）
        reference_date = ref_map[user]
        relative_timestamp = convert_to_relative_timestamp(real_date, reference_date)

        # 找出该条 tweet 上所有 symptom=1
        active_symptoms = []
        for col in SYMPTOM_COLUMNS:
            if col in row:
                try:
                    if float(row[col]) == 1.0:
                        active_symptoms.append(col)
                except:
                    pass

        if not active_symptoms:
            continue

        tweet_text = clean_tweet(row.get("content", ""))

        for sym in active_symptoms:
            timelines[user].append({
                "timestamp": relative_timestamp,
                "symptom": sym,
                "tweet": tweet_text
            })

    # ====== 生成输出 ======
    os.makedirs(output_dir, exist_ok=True)

    user_stats = []

    for user_id, records in tqdm(timelines.items(), desc="Writing symptom JSON & stats"):
        # 按时间排序
        records = sorted(records, key=lambda r: r["timestamp"])

        # 提取该用户所有出现过的 symptom
        symptom_set = sorted({r["symptom"] for r in records})

        # 写 JSON
        out_path = os.path.join(output_dir, f"{user_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "user_id": user_id,
                "symptoms": symptom_set,
                "timeline": records
            }, f, ensure_ascii=False, indent=2)

        # ===== 统计信息（完全复制你上午的风格） =====
        total_posts = len(records)
        if total_posts == 0:
            avg_per_cycle = 0
        else:
            timestamps = [r["timestamp"] for r in records]
            t_min, t_max = min(timestamps), max(timestamps)
            total_days = max(1, t_max - t_min)
            cycles = max(total_days / 90, 1)
            avg_per_cycle = total_posts / cycles

        user_stats.append({
            "user_id": user_id,
            "total_posts": total_posts,
            "avg_3_month_posts": avg_per_cycle
        })

    # 输出统计
    stats_df = pd.DataFrame(user_stats)
    stats_df.to_csv(os.path.join(output_dir, "user_symptom_statistics.csv"), index=False)

    print("Symptom relative timeline 生成完毕，输出目录：", output_dir)


def main_symptom():
    # ===== 修改成你的路径 =====
    input_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/symptom_only.csv"
    ref_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/anchor_depression_labeled.csv"
    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/stmhd_symptom_timeline"
    # ==========================

    ref_map = load_reference_dates(ref_csv, use_reference="label")

    build_symptom_relative_timeline(
        input_csv=input_csv,
        ref_map=ref_map,
        output_dir=output_dir,
        raw_dir="/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered",
        time_col="date"    # 如果原始 CSV 的列名不同（如 timestamp），就在这里改
    )




def main1():
    # ====== 修改你自己的路径 ======
    input_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/depression_life_events_extract.csv"
    ref_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/anchor_depression_labeled.csv"
    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/stmhd_timeline"

    # 使用 earliest_date 或 latest_date 作为脱敏基准
    use_reference = "label"
    # =================================

    # 1. 读取 reference 日历
    ref_map = load_reference_dates(ref_csv, use_reference)

def main_symptom():
    # ===== 修改成你的路径 =====
    input_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/deprofile_symptoms.csv"
    ref_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/anchor_depression_labeled.csv"
    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/stmhd_symptom_timeline"
    # ==========================

    ref_map = load_reference_dates(ref_csv, use_reference="label")

    build_symptom_relative_timeline(
        input_csv=input_csv,
        ref_map=ref_map,
        output_dir=output_dir,
        raw_dir="/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered",
        time_col="date"    # 如果原始 CSV 的列名不同（如 timestamp），就在这里改
    )




if __name__ == "__main__":
    main_symptom()
