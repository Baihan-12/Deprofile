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


def build_relative_timeline(input_csv, ref_map, output_dir):
    """
    根据 CSV + reference dates 生成脱敏后的 timeline（按用户分 JSON）
    并输出每个 user life_event=1 的帖子总数与平均每3个月周期的帖子数
    """
    df = pd.read_csv(input_csv)

    # 统一格式并排序
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")

    timelines = defaultdict(list)

    for _, row in tqdm(df.iterrows()):
        user = str(row["user_id"])

        if user not in ref_map:
            continue

        reference_date = ref_map[user]
        tweet_date = row["date"]

        # 计算相对天数（脱敏时间戳）
        relative_timestamp = convert_to_relative_timestamp(tweet_date, reference_date)

        # 只保留 model_output == 1 的内容
        if row.get("model_output", 0) == 1:
            timelines[user].append({
                "timestamp": relative_timestamp,
                "life_event": row.get("life_event", ""),
                "tweet": clean_tweet(row.get("tweet", ""))
            })

    # 输出 JSON
    os.makedirs(output_dir, exist_ok=True)

    # ======= 新增: 统计信息 =======
    user_stats = []

    for user_id, records in tqdm( timelines.items()):
        # 写 timeline
        out_path = os.path.join(output_dir, f"{user_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "user_id": user_id,
                "timeline": records
            }, f, ensure_ascii=False, indent=2)

        # 统计总数
        total_posts = len(records)

        if total_posts == 0:
            avg_per_cycle = 0
        else:
            timestamps = [r["timestamp"] for r in records]
            t_min = min(timestamps)
            t_max = max(timestamps)
            total_days = max(1, t_max - t_min)

            # 3个月 = 90天
            cycles = total_days / 90

            # 避免 cycles < 1 导致均值异常
            cycles = max(cycles, 1)

            avg_per_cycle = total_posts / cycles

        user_stats.append({
            "user_id": user_id,
            "total_posts": total_posts,
            "avg_3_month_posts": avg_per_cycle
        })

    # 输出统计 CSV
    stats_df = pd.DataFrame(user_stats)
    stats_df.to_csv(os.path.join(output_dir, "user_statistics.csv"), index=False)

    print("Timeline & 统计生成完毕，目录为：", output_dir)


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

    # 2. 生成脱敏 timeline
    build_relative_timeline(input_csv, ref_map, output_dir)

def main():
    # ===== 修改你的路径 =====
    input_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/depression_life_events_extract.csv"
  
    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/life_event_groups"
    # ========================

    extract_life_event_groups(input_csv, output_dir)
    print("全部 life_event 分类已完成，文件已输出到：", output_dir)


if __name__ == "__main__":
    main()
