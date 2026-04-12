import pandas as pd
import json
import os
from tqdm import tqdm
import re

LIFE_EVENTS_INFO = {
    "Health": {
        "category": "Health",
        "description": "Health-related life events",
        "subcategories": [
            "personal injury", "accident or illness", "got violently attacked (including sexual assault)",
            "became disabled", "mental illness", "recovery from mental health struggles",
            "major surgery", "Hospitalization", "Pregnancy", "Pregnancy loss & Abortion",
            "Menopause", "abuse (including sexual abuse)", "began to self-harm",
            "suicide attempt", "Substance Abuse and Addiction", "recovery from addiction",
            "loss of healthcare", "physical fitness milestone", "sex difficulties",
            "change in health of family member", "taking medicine"
        ]
    },
    "Financial": {
        "category": "Financial",
        "description": "Financial life events",
        "subcategories": [
            "change in financial state", "Loan", "home purchase", "car purchase",
            "other major purchase", "home improvement", "paid off debt",
            "major financial difficulty", "major financial gain", "claimed bankruptcy",
            "Foreclosure", "Mortgage", "personal property damaged or stolen"
        ]
    },
    "Relocation": {
        "category": "Relocation",
        "description": "Relocation and moving events",
        "subcategories": [
            "move within same town/city", "move to a different town/city",
            "move to a different state", "move to a different country",
            "move to a different country as a refugee",
            "became a permanent resident or citizen of a new country",
            "move out of parent's home", "move in with family",
            "family member moved into household", "family member moved out of household",
            "moved into assisted living", "lost home / became homeless", "major travel"
        ]
    },
    "Legal": {
        "category": "Legal",
        "description": "Legal issues and law-related events",
        "subcategories": [
            "got arrested", "lawsuit or legal action", "turned over power of attorney",
            "loss of driver's license / DUI", "went to jail or prison",
            "released from jail or prison", "Minor violations of the law"
        ]
    },
    "Relationships_Changes": {
        "category": "Relationships_Changes",
        "description": "Changes in relationships",
        "subcategories": [
            "began serious romantic relationship", "ended serious romantic relationship",
            "Engagement", "ended engagement", "Marriage", "Divorce",
            "marital separation", "marital reconciliation", "relationship became abusive",
            "serious argument with neighbor or friend", "change in number of arguments",
            "serious argument with family member or relative", "trouble with in-laws",
            "family betrayal", "parenting difficulties"
        ]
    },
    "New_Birth_in_Family": {
        "category": "New_Birth_in_Family",
        "description": "New family members",
        "subcategories": [
            "gain of new family member", "gave birth / became a parent",
            "adopted a child", "became a grandparent", "became a great-grandparent",
            "became an aunt/uncle"
        ]
    },
    "Death": {
        "category": "Death",
        "description": "Death of loved ones",
        "subcategories": [
            "death of spouse", "death of child", "death of parent", "death of pet",
            "death of a friend", "death of a loved one", "death of extended family member"
        ]
    },
    "Career": {
        "category": "Career",
        "description": "Career and work-related events",
        "subcategories": [
            "started a new job", "change in responsibilities at work",
            "change to a different line of work", "change in work hours or conditions",
            "business readjustment", "Promotion", "Demotion", "significant success at work",
            "troubles at work", "workplace discrimination or harassment",
            "voluntary job loss", "involuntary job loss",
            "became a business owner / entrepreneur", "Retirement",
            "unable to find work", "spouse begins or stops work"
        ]
    },
    "Education": {
        "category": "Education",
        "description": "Education-related events",
        "subcategories": [
            "begin or end school/college", "change in school/college",
            "left school (without graduating)", "denied entry into school",
            "obtained a certification", "Examination"
        ]
    },
    "Lifestyle_Change": {
        "category": "Lifestyle_Change",
        "description": "Changes in lifestyle and daily habits",
        "subcategories": [
            "change in physical habits", "change in responsibilities in personal life",
            "new pet", "joined the military", "returned to civilian life after military",
            "change in living conditions", "revision of personal habits",
            "change in recreation", "change in social activities", "vacation"
        ]
    },
    "Identity": {
        "category": "Identity",
        "description": "Identity-related changes",
        "subcategories": [
            "identified sexual preference", "identified gender", "came out as LGBTQ+",
            "gender transition", "change in political beliefs",
            "change in religious/spiritual beliefs or practices",
            "coming of age ceremony", "new sexual experience",
            "another major identity shift"
        ]
    },
    "Societal": {
        "category": "Societal",
        "description": "Societal and major world events",
        "subcategories": [
            "natural disaster", "Pandemic", "War",
            "major political event that had personal impact", "met a celebrity"
        ]
    }
}


def extract_life_events():
    """
    从所有用户的 STP.json 文件中提取有 life_events 标注的帖子
    如果一条帖子有多个 life_events，则保存多行
    """
    profiles_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
    base_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/stmhd_labels/depression/"
    output_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/life_events_extracted.csv"
    
    # 读取 profiles 获取用户 ID 顺序
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    all_records = []
    
    for user_id in tqdm(profiles.keys()):
        json_path = f"{base_dir}{user_id}/STP.json"
        
        if not os.path.exists(json_path):
            print(f"⚠️  文件不存在: {json_path}")
            continue
        
        # 读取 STP.json 文件
        with open(json_path, "r", encoding="utf-8") as f:
            tweets = json.load(f)

        
        count = 0
        for tweet in tweets:
            life_events = tweet.get('life_events', [])
            
            if not life_events or len(life_events) == 0:
                continue
            
            # 如果有多个 life_events，每个保存一行
            for event in life_events:
                record = {
                    'user_id': user_id,
                    'tweet': tweet.get('text', ''),
                    'date': tweet.get('timestamp_tweet', ''),
                    'tweet_id': tweet.get('tweet_id', ''),
                    'disorder_flag': tweet.get('disorder_flag', False),
                    'life_event': event
                }
                all_records.append(record)
                count += 1
        
        if count > 0:
            print(f"✅ 用户 {user_id}: 提取了 {count} 条 life_events 记录")
    
    # 保存为 CSV
    if len(all_records) > 0:
        result_df = pd.DataFrame(all_records)
        result_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n✅ 共提取 {len(all_records)} 条记录，已保存到 {output_path}")
    else:
        print("\n⚠️  没有找到任何 life_events 记录")


def create_life_event_prompts(output_path,num_rows=None):
    """
    读取 life_events_extracted.csv 的前 num_rows 行，生成 prompts
    
    参数:
        num_rows: 要处理的行数
        output_path: 输出的 JSON 文件路径
    """
    csv_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/life_events_extracted.csv"
    
    # 读取 CSV
    df = pd.read_csv(csv_path, encoding="utf-8")
    if num_rows:
        df = df.head(num_rows)
    
    system_prompt = """You are an assistant that analyzes social media posts with annotated life events. Your job is to determine whether the event actually happened to the poster. If yes, output keywords summarizing the event (e.g., "breakup", "illness", "moving"). If not, output "None". Always follow the required JSON format."""
    
    keywords_dict = {
        "Career": ["job_loss", "fired", "resigned", "quit", "promotion", "new_job", "career_change", "unemployment", "internship"],
        "Death": ["death", "passed_away", "bereavement", "loss"],
        "Education": ["graduation", "dropout", "exam", "admission", "failed_exam", "start_school", "transfer_school"],
        "Financial": ["debt", "bankruptcy", "financial_stress", "lost_money", "bonus", "salary_increase"],
        "Health": ["illness", "injury", "hospitalized", "surgery", "diagnosis", "chronic_condition", "covid"],
        "Identity": ["identity_change", "coming_out", "gender_transition", "self_discovery", "religious_change"],
        "Legal": ["lawsuit", "arrest", "fine", "legal_issue", "court_case", "divorce_filing"],
        "Lifestyle_Change": ["diet_change", "quit_smoking", "quit_drinking", "major_routine_change", "lifestyle_shift"],
        "New_Birth_in_Family": ["pregnancy", "childbirth", "new_baby", "expecting", "miscarriage"],
        "Relationships_Changes": ["breakup", "divorce", "engagement", "new_relationship", "argument", "relationship_issue"],
        "Relocation": ["moving", "immigration", "relocation", "leaving_home", "moving_out"],
        "Societal": ["political_event", "social_issue", "public_crisis", "disaster"]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows()):
            life_event = row['life_event']
            candidate_keywords = keywords_dict.get(life_event, [])
            
            user_prompt = f"""You will receive a social media post containing:
- a post ID,
- a life event label,
- a timestamp,
- and the post content.

Your tasks:
1. Determine whether the life event actually happened to the poster.
   - If it happened to the poster → process it.
   - If it refers to someone else, news, political commentary, or general opinions → output "None".

2. If the event concerns the poster, extract **one concise keyword** describing the event.
   You may refer to the following keyword suggestions for this specific life event label,
   but you are NOT restricted to this list. You may output any reasonable keyword.

Keyword suggestions for {life_event}:
{candidate_keywords}

3. Output a JSON dictionary with:
   - "id": the post ID
   - "label": the life event label or "None"
   - "time": the timestamp
   - "keyword": the extracted keyword or "None"

Examples:
Input:
id: 12345
Career: 2025-06-01: Trump is not a good president; he only works for his own money.
Output:
"None"

Input:
id: 12345
Relationships_Changes: 2020-03-03: Today I broke up with my boyfriend; he fell in love with someone else.
Output:
{{"id": "12345", "label": "Relationships_Changes", "time": "2020-03-03", "keyword": "breakup"}}

Now process this post:
id: {row['user_id']}
{life_event}: {row['date']}: {row['tweet']}"""
            
            prompt = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            json.dump(prompt, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✅ 已生成 {len(df)} 条 prompts，保存到 {output_path}")


def create_life_event_no_thinking(output_path,num_rows=None):
    """
    读取 life_events_extracted.csv 的前 num_rows 行，生成 prompts
    
    参数:
        num_rows: 要处理的行数
        output_path: 输出的 JSON 文件路径
    """
    csv_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/depression_life_events_extract.csv"
    
    # 读取 CSV
    df = pd.read_csv(csv_path, encoding="utf-8")
    if num_rows:
        df = df.head(num_rows)
    
    system_prompt = """Here is a post from a twitter user, please check if the given label is related to the post and if the user experiences the label themselves. please only output an int number"""
    
    
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows()):
            life_event = row['life_event']
            
            user_prompt = f"""/no_thinking Here is a post, please check if the label {life_event} is related to the post and if the user experiences the label {life_event} themselves.

Here is the detailed information about the label {life_event}:
{LIFE_EVENTS_INFO[life_event]}

Please strictly output an int in [-1,0,1] , no additional text:

-1 means the post is not related to the label at all, 0 means the post is related to the label but it is uncertain whether the label is experienced by the user, and 1 means the label is experienced by the user.

Here are the examples:

### Example -1
Label: Career
Post: I finally figured out why my phone battery drains so fast — it was the weather app running nonstop in the background.
Expected output: -1

### Example 0
Label: Career
Post: Thinking about whether I should switch jobs next year, but I’m still not sure what direction to take.
Expected output: 0

### Example 1
Label: Career
Post: I quit my job today. It was emotionally exhausting, but I finally did it.
Expected output: 1

ONLY output valid int, WITHOUT any extra characters or explanations outside the int. Here is the prompt:"""
            
            prompt = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt+row["tweet"]}
                ]
            }
            
            json.dump(prompt, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✅ 已生成 {len(df)} 条 prompts，保存到 {output_path}")

def clear_life_event_output(qwenoutput, save=False, maxlength=None):
    """
    清理 life event 的输出结果并保存到 CSV
    
    参数:
        qwenoutput: 模型输出文件路径
        save: 是否保存到 CSV，默认 False
        maxlength: 如果不保存，打印前 maxlength 条结果
    """
    csv_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/depression_life_events_extract.csv"
    
    with open(qwenoutput, "r", encoding="utf-8") as f:
        responses = f.readlines()
    
    # 清理每条响应
    cleaned_responses = []
    for i in tqdm(range(len(responses))):
        response = responses[i].strip()
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        response = json.loads(response)["response"].replace("\n", "")
        response = response.strip()
        cleaned_responses.append(response)
    df = pd.read_csv(csv_path, encoding="utf-8")
    if save:
        # 读取原始 CSV
        
        
        # keywords = []
        # for response in cleaned_responses:
        #     try:
        #         # 处理转义的双引号
        #         response = response.strip('"').replace('""', '"')
        #         data = json.loads(response)
        #         keyword = data['keyword']
        #         keywords.append(keyword)
        #     except Exception as e:
        #         print(f"解析错误: {e}")
        #         keywords.append(None)
        # for response in cleaned_responses:
        #     # 使用正则表达式提取 keyword 后面的值
        #     match = re.search(r'"keyword":\s*"([^"]*)"', response)
        #     if match:
        #         keyword = match.group(1)
        #         keywords.append(keyword)
        #     else:
        #         keywords.append(None)
        # cleaned_responses = keywords
        # 确保行数匹配
        if len(cleaned_responses) != len(df):
            print(f"⚠️  警告：响应数量 ({len(cleaned_responses)}) 与 CSV 行数 ({len(df)}) 不匹配")
            min_len = min(len(cleaned_responses), len(df))
            cleaned_responses = cleaned_responses[:min_len]
            df = df.head(min_len)
        
        # 添加清理后的结果到 CSV
        df['model_output'] = cleaned_responses
        
        # 保存更新后的 CSV
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"✅ 已将 {len(cleaned_responses)} 条结果保存到 {csv_path}")
    else:
        # 不保存，只打印
        if maxlength:
            cleaned_responses = cleaned_responses[:maxlength]
        
        for i in range(100):
            print(df.loc[i,"life_event"], df.loc[i,"tweet"], cleaned_responses[i+100])




def clean_and_split_life_events():
    """
    清理 life_events_extracted.csv 中 model_output 为 None 的行，
    保存清理后的文件，并按 user_id 分别保存
    """
    input_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/life_events_extracted.csv"
    cleaned_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/life_events_cleaned.csv"
    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_life_event"
    
    # 读取原始 CSV
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"原始文件长度: {len(df)}")
    
    # 删除 model_output 为 None 或 "None" 的行
    df_cleaned = df[df['model_output'].notna()]
    df_cleaned = df_cleaned[df_cleaned['model_output'] != 'None']
    df_cleaned = df_cleaned[df_cleaned['model_output'] != '"None"']
    
    # 保存清理后的文件
    df_cleaned.to_csv(cleaned_path, index=False, encoding="utf-8")
    print(f"清理后文件长度: {len(df_cleaned)}")
    print(f"✅ 已保存清理后的文件到 {cleaned_path}")
    
    # 统计用户数量和平均帖子数
    user_count = df_cleaned['user_id'].nunique()
    avg_posts = len(df_cleaned) / user_count if user_count > 0 else 0
    print(f"\n📊 统计信息:")
    print(f"   用户数量: {user_count}")
    print(f"   平均每用户帖子数: {avg_posts:.2f}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 按 user_id 分组并保存
    for user_id, group in df_cleaned.groupby('user_id'):
        output_path = f"{output_dir}/{user_id}.csv"
        group.to_csv(output_path, index=False, encoding="utf-8")
        # print(f"✅ 用户 {user_id}: {len(group)} 条记录 → {output_path}")
    
    print(f"\n✅ 共为 {user_count} 个用户创建了独立文件")






if __name__ == "__main__":
# 使用
    # extract_life_events()
    output_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/prompts/all_life_event_prompts.json"
    # create_life_event_prompts(output_path)  # 读取前100行

    qwenoutput = "/hpc_stor03/sjtu_home/baihan.li/deprofile/qwenoutput/all_life_event_prompts"
    clear_life_event_output(qwenoutput, save=True)

    # create_life_event_no_thinking(output_path)

    # 使用
    # clean_and_split_life_events()