import json
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
def age_to_range(age):
    """将年龄数字映射到年龄范围"""
    if age == 'Unknown' or age is None:
        return 'Unknown'
    
    age = int(age)
    if age <= 17:
        return '0-17'
    elif age <= 25:
        return '18-25'
    elif age <= 35:
        return '26-35'
    elif age <= 50:
        return '36-50'
    else:
        return '50+'

def add_profile_index_to_deprofiles():
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/deprofiles_main.json", "r") as f:
        data = json.load(f)
    new_data = {}
    for idx in tqdm(range(len(data))):
        new_data[str(10000+idx)[1:]] = data[idx]
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/deprofiles_main_index.json", "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"✅ 已更新 {len(data)} 个 depression patient profiles")


def check_candidate_number():
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/deprofiles_main_index.json", "r") as f:
        data = json.load(f)
    count = 0
    for idx, item in tqdm(data.items()):
        if len(item['candidate_id']) == 0:
            print(f"❌ CR[{idx}] 没有候选")
            count += 1
        else:
            print(f"✅ CR[{idx}] 有 {len(item['candidate_id'])} 个候选")
    print(f"✅ 共检查了 {len(data)} 个 depression patient profiles")
    print(f"❌ 没有候选的 CR 数量: {count}")


def resave_d4_dialogues():
    with open("/hpc_stor03/sjtu_home/baihan.li/AIAgent/datasets/d4/dialog_final_mapped.json", "r") as f:
        data = json.load(f)
    d4_profiles = {}
    for item in tqdm(data):
        idx = item["dialogId"]
        protrait = item["portrait"]
        protrait["depression_risk"] = item["dRisk"]
        protrait["suiside_risk"] = item["sRisk"]
        d4_profiles[idx] = protrait
        dialogues = []
        for dialogue in item["dialog"]:
            dialogues.append({"role": dialogue["role"], "content": dialogue["content"]})
        with open(os.path.join("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/d4_dialogues", idx+".json"), "w") as f:
            json.dump(dialogues, f, indent=4, ensure_ascii=False)
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/d4_profiles.json", "w") as f:
        json.dump(d4_profiles, f, indent=4, ensure_ascii=False)

def select_candidate_tweetid():
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/3_basic_info.json", "r") as f:
        basic_profiles = json.load(f)
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/cr_d4.json", "r") as f:
        cr_profiles = json.load(f)
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/0_en_ch_symp_pair.json", "r") as f:
        syp_pairs = json.load(f)
    # 为每个 CR profile 初始化 candidate_id 列表
    for cr_data in cr_profiles:
        cr_data['candidate_id'] = []
        cr_data["cr_id"] = str(cr_data["cr_id"])
        cr_data["d4_id"] = str(cr_data["d4_id"])
    
    match_count = 0
    
    for idx, cr_data in tqdm(enumerate(cr_profiles)):
        for basic_id, basic_data in basic_profiles.items():
            # 检查四个字段是否匹配
            fields_match = True
            
            # gender 匹配
            cr_gender = cr_data.get('gender', 'Unknown')
            basic_gender = basic_data.get('gender', 'Unknown')
            if cr_gender != 'Unknown' and basic_gender != 'Unknown' and cr_gender != basic_gender:
                fields_match = False
            
            # age 匹配 (将 cr 的数字年龄映射到范围)
            cr_age = age_to_range(cr_data.get('age', 'Unknown'))
            basic_age = basic_data.get('age', 'Unknown')
            if cr_age != 'Unknown' and basic_age != 'Unknown' and cr_age != basic_age:
                fields_match = False
            
            # marital_status 匹配
            cr_marital = cr_data.get('marital_status', 'Unknown')
            basic_marital = basic_data.get('marital_status', 'Unknown')
            if cr_marital != 'Unknown' and basic_marital != 'Unknown' and cr_marital != basic_marital:
                fields_match = False
            
            # work_status 匹配
            cr_work = cr_data.get('work_status', 'Unknown')
            basic_work = basic_data.get('work_status', 'Unknown')
            if cr_work != 'Unknown' and basic_work != 'Unknown' and cr_work != basic_work:
                fields_match = False
            
            if not fields_match:
                continue
            
            # 检查 big_five 是否存在
            if 'big_five' not in cr_data or 'big_five' not in basic_data:
                continue

            # 匹配 symptoms
            positive_symptoms = cr_data.get('positive_symptoms', [])
            negative_symptoms = cr_data.get('negative_symptoms', [])
            tweet_symptoms = basic_data.get('symptoms', False)
            if not tweet_symptoms:
                continue
            tweet_symptoms = [syp_pairs[symptom] for symptom in tweet_symptoms if symptom in syp_pairs]
            if len(tweet_symptoms) == 0:
                continue
            symp_similarity = 0
            for tweet_symptom in tweet_symptoms:
                if tweet_symptom in negative_symptoms:
                    fields_match = False
                    break
                if tweet_symptom in positive_symptoms:
                    symp_similarity += 1
            symp_similarity = symp_similarity / len(tweet_symptoms)
            if not fields_match or symp_similarity < 0.5:
                continue
            

            # 提取 big_five 向量
            traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            cr_vector = [cr_data['big_five'].get(trait, 0) for trait in traits]
            basic_vector = [basic_data['big_five'].get(trait, 0) for trait in traits]
            
            # 计算余弦相似度
            similarity = cosine_similarity([cr_vector], [basic_vector])[0][0]
            
            if similarity > 0.8:
                tmp_candidate_id = {"basic_id": basic_id, "similarity": similarity, "symp_similarity": symp_similarity}
                cr_profiles[idx]['candidate_id'].append(tmp_candidate_id)
                match_count += 1
        # print the candidate number
        # 候选按照两个similary之和排序
        cr_profiles[idx]['candidate_id'].sort(key=lambda x: x['similarity'] + x['symp_similarity'], reverse=True)
                # print(f"✅ 匹配: CR[{idx}] (d4_id:{cr_data.get('d4_id')}) <-> Basic {basic_id}, 相似度: {similarity:.4f}")
    
    # 保存更新后的 CR profiles
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/deprofiles_main.json", "w", encoding="utf-8") as f:
        json.dump(cr_profiles, f, ensure_ascii=False, indent=4)
    
    print(f"\n📊 共找到 {match_count} 对匹配")
    print(f"✅ 已更新 {cr_path}，添加了 candidate_id 字段")
    
    # 统计每个 CR profile 的候选数量
    for idx, cr_data in enumerate(cr_profiles):
        candidate_count = len(cr_data['candidate_id'])
        if candidate_count > 0:
            print(f"   CR[{idx}] (d4_id:{cr_data.get('d4_id')}): {candidate_count} 个候选")



def replace_none_with_Unknown():
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/3_basic_info.json", "r") as f:
        data = json.load(f)
    for idx, item in tqdm(data.items()):
        for key, value in item.items():
            if value is None:
                item[key] = "Unknown"
        data[idx] = item
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/3_basic_info.json", "w") as f:
        json.dump(data, f, indent=4)



def reselect_symptoms():
    " select all the symtoms and save in basic info"
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/3_basic_info.json", "r") as f:
        data = json.load(f)
    life_event_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/stmhd_life_event_timeline"
    symptom_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/stmhd_symptom_timeline"
    le_id = os.listdir(life_event_path)
    symptom_id = os.listdir(symptom_path)
    print(symptom_id[0])
    for idx, item in tqdm(data.items()):
 
        if str(idx)+".json" in le_id:
            item["life_events"] = True
        else:
            item["life_events"] = False
        if str(idx)+".json" in symptom_id:
            with open(os.path.join(symptom_path, idx+".json"), "r") as f:
                symptom = json.load(f)
            item["symptoms"] = symptom["symptoms"]
        else:
            item["symptoms"] = False
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/3_basic_info.json", "w") as f:
        json.dump(data, f, indent=4)

def split_dialogues_and_save_one_by_one():
    with open("/hpc_stor03/sjtu_home/baihan.li/AIAgent/datasets/d4_dialog.json", "r") as f:
        data = json.load(f)
    os.makedirs("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/d4_dialogues", exist_ok=True)
    for item in tqdm(data):
        idx = item["id"]
        conversation = item["conversations"]
        new_conversation = []
        for turn in conversation:
            if turn["from"] == "gpt":
                new_conversation.append({"role": "doctor", "content": turn["value"]})
            else:
                new_conversation.append({"role": "patient", "content": turn["value"]})
        with open(os.path.join("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/d4_dialogues", idx+".json"), "w") as f:
            json.dump(new_conversation, f, indent=4, ensure_ascii=False)
        
def split_client_dialogues_and_save_one_by_one():
    with open("/hpc_stor03/sjtu_home/baihan.li/AIAgent/datasets/CR.json", "r") as f:
        data = json.load(f)
    os.makedirs("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/cr_dialogues", exist_ok=True)
    for i in tqdm(range(len(data))):
        with open(os.path.join("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/cr_dialogues", str(i)+".json"), "w") as f:
            json.dump(data[i], f, indent=4, ensure_ascii=False)
if __name__ == "__main__":
    # check_candidate_number()
    resave_d4_dialogues()