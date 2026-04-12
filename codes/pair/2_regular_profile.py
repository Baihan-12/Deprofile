from tqdm import tqdm
import os
import csv
import re




def extract_gender(tweet):
    gender_keywords = [
        # 与性别角色相关的典型词汇
        r'\bmy wife\b', r'\bmy girlfriend\b', r'\bmy gf\b', r'\bmy lady\b', r'\bmy fiancée\b', r'\bI married her\b',r'\bmy husband\b', r'\bmy boyfriend\b', r'\bmy bf\b', r'\bmy fiancé\b', r'\bI married him\b'
    ]
    for pattern in gender_keywords:
        if re.search(pattern, tweet, re.IGNORECASE):
            return {"gender_flag": 1}
    return {"gender_flag": 0}

def extract_age(text):
    # 定义扩展的正则表达式模式，增加更多关于生日、未来年龄和模糊表达的匹配
    age_patterns = [
        # 显式年龄表达式
        r'(?i)\b(I am|I\'m|I am turning|I will be|I will turn|I\'ll turn|I am about to turn)\s*(\d{1,3})\s*years? old\b',  # I am X years old / I will turn X / I am about to turn X
        r'(?i)\b(I am|I\'m)\s*(\d{1,3})(?!%)\b',  # I’m 25
        r'(?i)\b(I was)? born in (\d{4})\b',  # 出生年份
        r'(?i)\b(I was)? born on (\d{1,2}\/\d{1,2}\/\d{4}|\d{1,2}\s*[A-Za-z]+\s*\d{4})\b',  # 出生日期（如 12/25/1990 或 25 December 1990）
        r'(?i)\b(\d{1,3})\s*to\s*(\d{1,3})\s*years?\s*old\b',  # 年龄范围，如 "20 to 30 years old"
        r'(?i)\bIn my (\d{1,2})s\b',  # "In my 30s"
        r'(?i)\bI\'ve been (\d{1,3})\s*(years?\s*old)?\s*for\b',  # 表达当前年龄 (例如 I’ve been 25 years old for a year)
        
        
        # 时间跨度和生日相关
        r'(?i)\bI just turned (\d{1,3})\b',  # “I just turned X”
        r'(?i)\bI turned (\d{1,3})\s*(last|this)\s*(week|month|year)\b',  # “I turned X last week/month/year”
        r'(?i)\bI turned (\d{1,3})\s*on\s*my\s*birthday\b',  # “I turned X on my birthday”
        r'(?i)\bI am (\d{1,3})\s*years? old\s*this\s*year\b',  # “I am X years old this year”
        r'(?i)\b(I am)? between (\d{1,3})\s*and\s*(\d{1,3})\s*years?\s*old\b',  # "I am between X and Y years old"
        r'(?i)\bI will be (\d{1,3})\s*years?\s*old\s*(next\s*year|next\s*month|next\s*week|next\s*day)?\b',  # "I will be X years old next year"
        r'(?i)\b(I will be)? (\d{1,3})\s*(years?\s*old)?\s*(next\s*year|next\s*month|next\s*week|next\s*day)\b',
        
        # 未来年龄表达式
        r'(?i)\b(I\'m going to turn)? (\d{1,3})\s*next\s*(today|tomorrow|week|month|year)\b',  # “I’m going to turn X next month”
        r'(?i)\bI\'ll turn (\d{1,3})\s*next\s*(today|tomorrow|week|month|year)\b',  # “I’ll turn X next week/month”
        r'(?i)\bI will turn (\d{1,3})\s*next\s*(today|tomorrow|week|month|year)\b',  # “I will turn X next week/month”
        
        # 黄金生日表达式
        r'(?i)\bI am having my golden birthday today(, turning (\d{1,3})?)\b',  # "golden birthday"
        r'(?i)\b\d+\s*.*\bbirthday\b',
        
        # 模糊的过去年龄
        r'(?i)\bHad just turned (\d{1,3})\b',  # "Had just turned X"
        r'(?i)\bI was (\d{1,3})\s*(years?\s*old)?\s*last\s*(year|week|month)\b',  # "I was X years old last year/month"
        
        # 混合表达方式，推测当前年龄
        r'(?i)\b(I)? will be (\d{1,3})\s*in\s*(\d{1,4})\s*(years?|months?|weeks?|days?)\b',  # "I will be X in Y years"
    ]

    for pattern in age_patterns:
        if re.search(pattern, text):
            return {"age_flag": 1}
    return {"age_flag": 0}                   


def extract_marital_status(text):
    marriage_patterns = {
         r"(?i)\b(married|marry|spouse|wife|husband|wedding|partner|tied the knot|marital|marriage|my better half)\b",
         r"(?i)\b(single|unmarried|bachelor|spinster|no boyfriend|no girlfriend|flying solo|living alone|not seeing anyone)\b",
         r"(?i)\b(divorced|divorce|ex-husband|exhusband|ex-wife|exwife|separated|divorcee|split up|ended our marriage|custody battle|amicable separation)\b",
         r"(?i)\b(widowed|widower|widow|lost my wife|lost my husband|late spouse|passed away|grieving my partner)\b"
    }
    for pattern in marriage_patterns:
        if re.search(pattern, text):
            return {"marital_flag": 1}
    return {"marital_flag": 0}


def extract_work_status(text):
    work_patterns = {
         r"(?i)\b(working|employed|full-time|part-time|freelancer|consultant|self-employed|entrepreneur|running a business|active in the workforce|intern|engaged in work|on a job|work|job|office|hired|employment|workplace)\b",
         r"(?i)\b(unemployed|jobless|looking for a job|seeking employment|out of work|laid off|fired|between jobs|job hunting|without a job|not working|unemployed|unemployment|unhired|job seeker|without employment|displaced worker)\b",
         r"(?i)\b(retired|pensioner|retirement|no longer working|former employee|post-career|pension|senior citizen|retire|retiring|post-retirement)\b",
         r"(?i)\b(student|studying|university|school|college|pursuing degree|enrolled|in class|on campus|taking courses|academic break)\b"
    }
    for pattern in work_patterns:
        if re.search(pattern, text):
            return {"work_flag": 1}
    return {"work_flag": 0}


def main():
    base_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered/"

    # 遍历文件夹下所有 CSV 文件
    csv_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.endswith(".csv")
    ]

    print(f"🔍 找到 {len(csv_files)} 个 CSV 文件，将依次处理...\n")

    for file_path in tqdm(csv_files, desc="Processing CSV files", ncols=90):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # 确保新增列存在
            new_fields = ["gender_flag", "age_flag", "marital_flag", "work_flag"]
            for nf in new_fields:
                if nf not in fieldnames:
                    fieldnames.append(nf)

            flagged_rows = []

            # 处理每条推文
            for row in rows:
                tweet = row["tweet"]

                gender_info = extract_gender(tweet)
                age_info = extract_age(tweet)
                marital_info = extract_marital_status(tweet)
                work_info = extract_work_status(tweet)

                row.update(gender_info)
                row.update(age_info)
                row.update(marital_info)
                row.update(work_info)

                if any([row["gender_flag"], row["age_flag"], row["marital_flag"], row["work_flag"]]):
                    flagged_rows.append(row)

            # 写回文件（原地更新）
            with open(file_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            if flagged_rows:
                tqdm.write(f"✅ {os.path.basename(file_path)} 标记 {len(flagged_rows)} 条")

        except Exception as e:
            tqdm.write(f"⚠️ 处理 {file_path} 时出错: {e}")

    print("\n🎉 所有文件处理完成！")
        


if __name__ == "__main__":
    main()