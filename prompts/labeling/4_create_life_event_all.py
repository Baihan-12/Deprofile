import os
import json
import pandas as pd

ROOT = "/hpc_stor03/sjtu_home/baihan.li/deprofile/stmhd_labels/depression"

TARGET_PREFIXES = [
    'Career', 'Death', 'Education', 'Financial','Identity',
    'Legal', 'Lifestyle_Change', 'New_Birth_in_Family',
    'Relationships_Changes', 'Relocation', 'Societal'
]

OUTPUT_CSV = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/depression_life_events_extract.csv"

rows = []   # pandas 的中间存储

for user_id in os.listdir(ROOT):
    user_path = os.path.join(ROOT, user_id)
    detail_dir = os.path.join(user_path, "detail")

    if not os.path.isdir(detail_dir):
        continue

    for filename in os.listdir(detail_dir):
        if not any(filename.startswith(prefix) for prefix in TARGET_PREFIXES):
            continue

        life_event = filename.split(".")[0]
        json_path = os.path.join(detail_dir, filename)

        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
        except:
            continue

        if isinstance(data, dict):
            data = [data]

        for item in data:
            rows.append({
                "user_id": user_id,
                "tweet": item.get("text", ""),
                "date": item.get("timestamp_tweet", ""),
                "tweet_id": item.get("tweet_id", ""),
                "disorder_flag": item.get("disorder_flag", ""),
                "life_event": life_event,
                "model_output": life_event   # 你暂时这么设
            })

# 用 pandas 输出
df = pd.DataFrame(rows)
print("maximum length of files:", len(df))
df.to_csv(OUTPUT_CSV, index=False)

print("Saved to:", OUTPUT_CSV)

