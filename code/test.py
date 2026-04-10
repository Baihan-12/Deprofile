import json
import random

import pandas as pd
from tqdm import tqdm

from agent import TimelineAgent
from repo_paths import DATA_DIR
from utils import load_clinical_dialogues
# 按照性别/年龄/婚姻状况/工作状况/筛选300个profile, check 其candidate， 保存3月平均tweet 数目在20-50的第一个candidate

# load the data

def change_age(age):
    if age < 18:
        return "0-17"
    elif age < 25:
        return "18-25"
    elif age < 35:
        return "26-35"
    elif age < 50:
        return "36-50"
    elif age > 50:
        return "50+"
    else:
        return "Unknown"

def complete_profiles():
    with open(DATA_DIR / "d4_profiles.json", "r", encoding="utf-8") as f:
        d4_profiles = json.load(f)
    with open(DATA_DIR / "deprofiles_main_index.json", "r", encoding="utf-8") as f:
        deprofiles = json.load(f)
    complete_profiles = {}
    for key, value in tqdm(deprofiles.items()):
        d4_id = value["d4_id"]
        if d4_id not in d4_profiles:
            print(f"d4_id {d4_id} not in d4_profiles")
            continue
        value["depression_risk"] = d4_profiles[d4_id]["depression_risk"]
        value["suiside_risk"] = d4_profiles[d4_id]["suiside_risk"]
        complete_profiles[key] = value
    with open(DATA_DIR / "deprofiles_complete_index.json", "w", encoding="utf-8") as f:
        json.dump(complete_profiles, f, indent=4, ensure_ascii=False)


def main():# filter the data
    profile_path = DATA_DIR / "deprofiles_complete_index.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    save_profiles = {}
    
    gender = ["M", "F", "Unknown"]
    # ['0-17', '18-25', '26-35', '36-50', '50+', 'Unknown']
    age = ["0-17", "18-25", "26-35", "36-50", "50+","Unknown"]
    occupation = ['employed','unemployed','student','retired','Unknown']
    # ['married','single','divorced','widowed','Unknown']
    marital_status = ["married","single","divorced","widowed","Unknown"]

    for g in gender:
        for a in age:
            for o in occupation:
                for m in marital_status:
                    save_profiles[f"{g}_{a}_{o}_{m}"] = []
    for user_id, profile in tqdm(profiles.items()):

        
        save_profiles[f"{profile['gender']}_{change_age(profile['age'])}_{profile['work_status']}_{profile['marital_status']}"].append(user_id)
    
    count = 0
    for value in save_profiles.values():
        count += len(value)
    print(count)
    
    with open(DATA_DIR / "class_profiles.json", "w", encoding="utf-8") as f:
        json.dump(save_profiles, f, indent=4, ensure_ascii=False)
    
    # select 15% of the profiles for each class then only save the first candidate id have 10-30 timeline life events or symptoms
    select_profiles = {}

    statistic_life_events = pd.read_csv(DATA_DIR / "stmhd_life_event_timeline/user_statistics.csv")
    statistic_symptoms = pd.read_csv(DATA_DIR / "stmhd_symptom_timeline/user_symptom_statistics.csv")
    for g in gender:
        for a in age:
            for o in occupation:
                for m in marital_status:
                    if len(save_profiles[f"{g}_{a}_{o}_{m}"]) == 0:
                        continue
                    tmp_profiles = {}
                    maximum_num = int(len(save_profiles[f"{g}_{a}_{o}_{m}"]) * 0.2)+1
                    tmp_num = 0
                    count_num = 0
            
                    while tmp_num < maximum_num:
                        user_id = save_profiles[f"{g}_{a}_{o}_{m}"][tmp_num]
                        profile = profiles[user_id]
                        if len(profile["candidate_id"]) > 0:
                            for candidate in profile["candidate_id"]:
                                candidate_id = int(candidate["basic_id"])
                                # print(type(statistic_life_events["user_id"].values[0]))
                                if candidate_id not in statistic_life_events["user_id"].values or candidate_id not in statistic_symptoms["user_id"].values:
                                    print("not in")
                                    continue
                                # print(statistic_life_events[statistic_life_events["user_id"] == candidate_id]["avg_3_month_posts"].values[0])
                                # print(statistic_symptoms[statistic_symptoms["user_id"] == candidate_id]["avg_3_month_posts"].values[0])
                                if statistic_life_events[statistic_life_events["user_id"] == candidate_id]["avg_3_month_posts"].values[0] >= 10 and statistic_life_events[statistic_life_events["user_id"] == candidate_id]["avg_3_month_posts"].values[0] <= 30 and statistic_symptoms[statistic_symptoms["user_id"] == candidate_id]["avg_3_month_posts"].values[0] >= 10 and statistic_symptoms[statistic_symptoms["user_id"] == candidate_id]["avg_3_month_posts"].values[0] <= 30:
                                    profiles[user_id]["candidate_id"] = [candidate]
                                    tmp_profiles[user_id] = profiles[user_id]
                                    count_num += 1
                                    break
                        tmp_num += 1
                        # print(tmp_profiles)
                    select_profiles.update(tmp_profiles)
    print(len(select_profiles))
    with open(DATA_DIR / "main_select_profiles.json", "w", encoding="utf-8") as f:
        json.dump(select_profiles, f, indent=4, ensure_ascii=False)

def main2():
    profile_path = DATA_DIR / "deprofiles_complete_index.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    save_profiles = {}
    
    gender = ["M", "F", "Unknown"]
    # ['0-17', '18-25', '26-35', '36-50', '50+', 'Unknown']
    age = ["0-17", "18-25", "26-35", "36-50", "50+","Unknown"]
    occupation = ['employed','unemployed','student','retired','Unknown']
    # ['married','single','divorced','widowed','Unknown']
    marital_status = ["married","single","divorced","widowed","Unknown"]

    with open(DATA_DIR / "class_profiles.json", "r", encoding="utf-8") as f:
        save_profiles = json.load(f)
    # ---------------------------------------------------------
    # 性能优化：预处理 CSV 数据
    # 在循环里查 Pandas DataFrame 非常慢，转成字典查询是 O(1) 的复杂度
    # ---------------------------------------------------------
    print("Loading and indexing statistics...")
    stat_life_events = pd.read_csv(DATA_DIR / "stmhd_life_event_timeline/user_statistics.csv")
    stat_symptoms = pd.read_csv(DATA_DIR / "stmhd_symptom_timeline/user_symptom_statistics.csv")

    # 制作查找表： {candidate_id: avg_3_month_posts}
    # 只有在该 id 存在于表中时才存入
    life_event_dict = dict(zip(stat_life_events["user_id"], stat_life_events["avg_3_month_posts"]))
    symptom_dict = dict(zip(stat_symptoms["user_id"], stat_symptoms["avg_3_month_posts"]))
    
    # 用于防止不同 User 抽到同一个 candidate_id (如果有这种可能性的话)
    used_candidate_ids = set() 
    
    select_profiles = {}
    
    print("Selecting profiles...")
    for g in gender:
        for a in age:
            for o in occupation:
                for m in marital_status:
                    key = f"{g}_{a}_{o}_{m}"
                    user_list = save_profiles[key]
                    
                    if len(user_list) == 0:
                        continue

                    # 2. 随机化：打乱当前分类下的用户列表
                    random.shuffle(user_list)

                    # 计算目标数量 (20%)
                    target_count = int(len(user_list) * 0.2) + 1
                    current_count = 0
                    
                    # 3. 修改循环逻辑：遍历打乱后的列表
                    for user_id in user_list:
                        # 如果凑够了数量，就停止当前分类的循环
                        if current_count >= target_count:
                            break
                            
                        profile = profiles[user_id]
                        
                        if len(profile.get("candidate_id", [])) > 0:
                            # 随机化：如果一个用户有多个 candidate，也随机打乱一下顺序
                            candidates = profile["candidate_id"]
                            random.shuffle(candidates)
                            
                            found_valid_candidate = False
                            
                            for candidate in candidates:
                                try:
                                    candidate_id = int(candidate["basic_id"])
                                except (ValueError, KeyError):
                                    continue

                                # 检查是否重复抽取（可选）
                                if candidate_id in used_candidate_ids:
                                    continue

                                # 使用字典快速查找
                                if candidate_id not in life_event_dict or candidate_id not in symptom_dict:
                                    # print("not in") # 建议注释掉，否则输出太多
                                    continue
                                
                                val_life = life_event_dict[candidate_id]
                                val_symp = symptom_dict[candidate_id]

                                # 判断条件
                                if (10 <= val_life <= 30) and (10 <= val_symp <= 30):
                                    # 找到了符合条件的 candidate
                                    # 创建一个新的 profile 对象，避免修改原始 profiles 里的引用
                                    import copy
                                    new_profile = copy.deepcopy(profiles[user_id])
                                    new_profile["candidate_id"] = [candidate] # 只保留这一个
                                    
                                    select_profiles[user_id] = new_profile
                                    used_candidate_ids.add(candidate_id) # 记录已使用
                                    
                                    current_count += 1
                                    found_valid_candidate = True
                                    break # 找到一个合法的 candidate 后，跳出 candidate 循环，处理下一个 user
                            
                            # 注意：这里不需要 else，逻辑都在 break 里处理了

    print(f"Total selected profiles: {len(select_profiles)}")
    with open(DATA_DIR / "main_select_profiles_2.json", "w", encoding="utf-8") as f:
        json.dump(select_profiles, f, indent=4, ensure_ascii=False)

TEST_PROFILE = {
        "cr_id": "0",
        "d4_id": "971",
        "age": 19,
        "gender": "F",
        "marital_status": "single",
        "work_status": "student",
        "big_five": {
            "Openness": 6,
            "Conscientiousness": 5,
            "Extraversion": 3,
            "Agreeableness": 7,
            "Neuroticism": 2
        },
        "candidate_id": [
            {
                "basic_id": "837785700460265472",
                "similarity": 0.902693677739844,
                "symp_similarity": 1.0
            },
            {
                "basic_id": "173367742",
                "similarity": 0.8880416016036986,
                "symp_similarity": 1.0
            },
            {
                "basic_id": "1323237388211023872",
                "similarity": 0.807393770514232,
                "symp_similarity": 1.0
            },
            {
                "basic_id": "1249650669943955457",
                "similarity": 0.8070906378696021,
                "symp_similarity": 0.5
            }
        ],
        "positive_symptoms": [
            "任务-自杀-存在自杀倾向",
            "闲聊-自我表露-对事物的情绪",
            "任务-社会功能-避免从亲友处得到支持",
            "任务-躯体症状-躯体不适",
            "任务-精神状态-缺乏自信",
            "任务-兴趣-兴趣丧失超过两周",
            "闲聊-提供信息-主动提供相关信息",
            "任务-社会功能-学习工作存在困难",
            "任务-自杀-有无望感",
            "闲聊-自我表露-抱怨自我",
            "任务-躯体症状-运动性激越",
            "任务-精神状态-疲倦",
            "任务-兴趣-兴趣丧失",
            "任务-兴趣-范围-过去爱好"
        ],
        "negative_symptoms": [
            "任务-情绪-早晚差异",
            "任务-食欲-食欲存在问题",
            "任务-自杀-自我价值感低",
            "任务-情绪-情绪低落",
            "任务-睡眠-多梦",
            "任务-睡眠-存在睡眠问题",
            "任务-自杀-存在自杀行为"
        ],
        "summation": "来访者持续两周以上对事物缺乏兴趣、情绪烦躁、有无力感，另外来访者反映会有头痛，睡眠和食欲一般，有自杀的想法，会有无价值感。综上判断来访者抑郁程度为中度。因来访者反映情绪持续烦躁，需进一步进行双向障碍的排查。"
    }

if __name__ == "__main__":
    # timeline_agent = TimelineAgent(TEST_PROFILE, 0, "symptom")
    # timeline = timeline_agent.get_cut_timeline(max_events_num = 50)
    # print(timeline)
    # complete_profiles()
    main2()
    