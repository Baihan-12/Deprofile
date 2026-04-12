import pandas as pd
import os
import json
import re
from tqdm import tqdm
from collections import Counter
from statistics import mean
import ast

def collect_flagged_tweets(csv_path, label):
    """
    从给定 CSV 文件中收集某个 label_flag=True 的帖子及其时间。
    返回一个列表，每个元素为 'date tweet'。
    
    参数:
        csv_path: 含有 *_flag 的 CSV 文件路径
        label: 要提取的标签，如 'gender'、'age'、'marital'、'work'
    """
    results = []
    flag_col = f"{label}_flag"

    df = pd.read_csv(csv_path, encoding="utf-8")
    if flag_col not in df.columns:
        print(f"❌ 从 {csv_path} 中找不到 {label}_flag 列")
        return results
    for _, row in df.iterrows():
        if row[flag_col]:
            results.append(f"{row['date']} {row['tweet']}")
        

    # print(f"✅ 从 {csv_path} 中提取到 {len(results)} 条 {label}_flag=True 的帖子")
    return results

def create_judge_prompts(output_dir, system_prompt, user_prompt,label, max_length=None):

    home_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered/"
    profiles = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
    with open(profiles, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    output_path = f"{output_dir}/{label}_judge_prompts.json"
    
    for key, value in tqdm(profiles.items()):
        tweets = collect_flagged_tweets(f"{home_dir}/{key}.csv", label)
        
        if len(tweets) == 0:
            profiles[key][label] = "Unknown"
            continue
        else:
            tweets = " ".join(tweets)
            if len(tweets) > max_length:
                tweets = tweets[:max_length]
            tmp = {"messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt+"id:"+key+" tweets:"+tweets},],}
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(tmp, f, ensure_ascii=False)
                f.write("\n")
    # with open( profiles, "w", encoding="utf-8") as f:
    #     json.dump(profiles, f, ensure_ascii=False, indent=4)

def extract_first_n_lines(json_filename, n=50):
    """
    读取 JSON 文件的前 n 行，输出到同一文件夹下的 test 文件
    
    参数:
        json_filename: JSON 文件路径
        n: 要提取的行数，默认为 50
    
    返回:
        输出文件的路径
    """
    # 检查文件是否存在
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"文件不存在: {json_filename}")
    
    # 获取文件路径信息
    dir_path = os.path.dirname(json_filename)
    filename = os.path.basename(json_filename)
    name, ext = os.path.splitext(filename)
    
    # 生成输出文件名
    output_filename = os.path.join(dir_path, f"{name}_test{ext}")
    
    # 读取前 n 行并写入新文件
    with open(json_filename, 'r', encoding='utf-8') as infile:
        lines = []
        for i, line in enumerate(infile):
            if i >= n:
                break
            lines.append(line)
    
    # 写入输出文件
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.writelines(lines)
    
    print(f"已提取前 {len(lines)} 行到: {output_filename}")
    return output_filename


def create_profile_file():
    home_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered/"
    output_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
    
    profiles = {}
    for f in os.listdir(home_dir):
        if f.endswith(".csv"):
            idx = os.path.splitext(f)[0]
            profiles[idx] = {"gender": None,  "age": None, "marital_status": None, "work_status": None}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=4)


def create_bf_prompts(output_dir, system_prompt, user_prompt, max_tweets=1500, max_length=None):
    """
    为所有用户创建判断 prompts，使用所有 tweets，每100条一组
    
    参数:
        output_dir: 输出目录
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        max_tweets: 每个用户最多处理的 tweets 数量，默认 1500
        max_length: 每组 tweets 的最大字符长度限制（可选）
    """
    home_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered/"
    profiles = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
    
    with open(profiles, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    output_path = f"{output_dir}/bf_prompts.json"
    
    for key in tqdm(profiles.keys()):
        csv_path = f"{home_dir}/{key}.csv"
        
        if not os.path.exists(csv_path):
            print(f"⚠️  文件不存在: {csv_path}")
            continue
        
        # 读取 CSV 并收集 tweets
        df = pd.read_csv(csv_path, encoding="utf-8")
        all_tweets = []
        for _, row in df.iterrows():
            if len(all_tweets) >= max_tweets:
                break
            all_tweets.append(f"{row['date']} {row['tweet']}")
        
        if len(all_tweets) == 0:
            print(f"⚠️  用户 {key} 没有 tweets")
            continue
        
        # 每100条分组
        for i in range(0, len(all_tweets), 100):
            group = all_tweets[i:i+100]
            tweets_group = " ".join(group)
            
            if max_length and len(tweets_group) > max_length:
                tweets_group = tweets_group[:max_length]
            
            group_idx = i // 100 + 1
            tmp = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prompt} id:{key} tweets:{tweets_group}"}
                ]
            }
            
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(tmp, f, ensure_ascii=False)
                f.write("\n")
        
        print(f"✅ 用户 {key}: {len(all_tweets)} 条帖子，分为 {(len(all_tweets)-1)//100 + 1} 组")
    
    print(f"✅ 所有 prompts 已保存到 {output_path}")


def statistics_bf(max_tweets=1500):
    """
    统计每个用户按100条分组后的 prompt 数量，并保存到 profile 中
    
    参数:
        max_tweets: 每个用户最多处理的 tweets 数量，默认 1500
    """
    home_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered/"
    profiles_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
    
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    for key in profiles.keys():
        csv_path = f"{home_dir}/{key}.csv"
        
        if not os.path.exists(csv_path):
            profiles[key]['prompt_count'] = 0
            continue
        
        # 读取 CSV 并收集 tweets
        df = pd.read_csv(csv_path, encoding="utf-8")
        all_tweets = []
        for _, row in df.iterrows():
            if len(all_tweets) >= max_tweets:
                break
            all_tweets.append(f"{row['date']} {row['tweet']}")
        
        # 计算分组数量（每100条一组）
        prompt_count = (len(all_tweets) + 99) // 100  # 向上取整
        profiles[key]['prompt_count'] = prompt_count
        
        print(f"用户 {key}: {len(all_tweets)} 条 tweets → {prompt_count} 个 prompts")
    
    # 保存更新后的 profiles
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ 已更新 profile，添加了 prompt_count 字段")
    
    # 统计总数
    total_prompts = sum(p.get('prompt_count', 0) for p in profiles.values())
    print(f"📊 总共生成 {total_prompts} 个 prompts")


def clear_text(response):
    """
    清理单行 response 文本
    
    参数:
        response: 原始响应字符串
    
    返回:
        清理后的字符串
    """
    response = response.strip()
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    response = json.loads(response)["response"].replace("\n", "")
    response = response.strip()
    response = re.sub(r"['\"]", "", response)
    return response



def parse_bf_results():
    """
    从 bf_prompts_merged 文件中按照每个用户的 prompt_count 读取结果，
    合并多个 prompt 的大五人格评分并保存到 profile 中
    """
    merged_file = "/hpc_stor03/sjtu_home/baihan.li/deprofile/qwenoutput/bf_prompts_merged"
    profiles_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
    
    # 读取 profiles
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    # 读取所有输出结果
    with open(merged_file, "r", encoding="utf-8") as f:
        results = f.readlines()
    
    result_idx = 0
    
    for key in profiles.keys():
        prompt_count = profiles[key].get('prompt_count', 0)
        
        if prompt_count == 0:
            print(f"⚠️  用户 {key} 没有 prompts，跳过")
            continue
        
        # 收集该用户的所有结果
        user_results = []
        for i in range(prompt_count):
            if result_idx >= len(results):
                print(f"⚠️  结果文件行数不足，用户 {key} 缺少数据")
                break
            
            line = results[result_idx].strip()
            line = clear_text(line)
            # print(line)
            result_idx += 1
            
            # 解析结果，提取大五人格分数
            try:
                # 尝试提取包含大五人格的字典部分
                match = re.search(r'\{[^{}]*?(Openness|openness)[^{}]*?\}', line)
                if match:
                    json_str = match.group(0)
                    
                    # 标准化键名为首字母大写
                    json_str = re.sub(r'\b(openness|Openness)\b', '"Openness"', json_str)
                    json_str = re.sub(r'\b(conscientiousness|Conscientiousness)\b', '"Conscientiousness"', json_str)
                    json_str = re.sub(r'\b(extraversion|Extraversion)\b', '"Extraversion"', json_str)
                    json_str = re.sub(r'\b(agreeableness|Agreeableness)\b', '"Agreeableness"', json_str)
                    json_str = re.sub(r'\b(neuroticism|Neuroticism)\b', '"Neuroticism"', json_str)
                    
                    # 替换单引号为双引号
                    json_str = json_str.replace("'", '"')
                    
                    try:
                        bf_scores = json.loads(json_str)
                    except json.JSONDecodeError:
                        # 如果 JSON 解析失败，尝试用 ast.literal_eval（处理 Python 字典格式）
                        bf_scores = ast.literal_eval(match.group(0))
                    
                    user_results.append(bf_scores)
                else:
                    print(f"⚠️  无法解析用户 {key} 的第 {i+1} 个结果: {line[:100]}")
            except Exception as e:
                print(f"⚠️  解析错误 (用户 {key}, 第 {i+1} 个结果): {e}")
                print(f"   原始内容: {line[:200]}")
                continue
        
        if len(user_results) == 0:
            print(f"⚠️  用户 {key} 没有有效结果")
            continue
        
        # 合并结果
        if len(user_results) == 1:
            # 只有一个结果，直接使用
            final_scores = user_results[0]
        else:
            # 多个结果，按照策略合并
            final_scores = {}
            traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
            
            for trait in traits:
                scores = [r[trait] for r in user_results if trait in r]
                
                if len(scores) == 0:
                    continue
                
                # 统计每个分数出现的次数
                counter = Counter(scores)
                max_count = max(counter.values())
                most_common = [score for score, count in counter.items() if count == max_count]
                
                if len(most_common) == 1:
                    # 有明确的多数
                    final_scores[trait] = most_common[0]
                else:
                    # 有多个分数出现次数相同，取平均
                    final_scores[trait] = round(mean(most_common))

        
        # 保存到 profile
        profiles[key]['big_five'] = final_scores
        print(f"✅ 用户 {key}: {prompt_count} 个 prompts → {final_scores}")
        # break
    
    # 保存更新后的 profiles
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ 已更新所有用户的大五人格评分到 profile")


def clear_label_output(label_name, qwenoutput, save=False, maxlength = None):
    with open(qwenoutput, "r", encoding="utf-8") as f:
        responses = f.readlines()
    for i in tqdm(range(len(responses))):
        responses[i] = responses[i].strip()
        responses[i] = re.sub(r"<think>.*?</think>", "", responses[i], flags=re.DOTALL).strip()
        responses[i] = json.loads(responses[i])["response"].replace("\n", "")
        responses[i] = responses[i].strip()
        responses[i] = re.sub(r"['\"]", "", responses[i])
    if save:
        profiles = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
        with open(profiles, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        for i in tqdm(range(len(responses))):
            item = responses[i]
            item = item.strip('{}')
            # 分割 ID 和 label
            user_id, label = item.split(':', 1)
            user_id = user_id.strip()
            label = label.strip()
            
            # 添加到 profiles（如果已存在则更新 label 字段）
            if user_id not in profiles:
                user_id = user_id[2:]
            if user_id not in profiles:
                continue
            profiles[user_id][label_name] = label
        new_profiles = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
        with open(new_profiles, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=4)
    else:
        if maxlength:
            print(responses[:maxlength])
        else:
            print(responses)

def print_flag_true(label, length):
    home_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered/"
    profiles = "/hpc_stor03/sjtu_home/baihan.li/deprofile/jsonfile/3_basic_info.json"
    with open(profiles, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    for key in profiles.keys():
        if length == 0:
            break
        tweets = collect_flagged_tweets(f"{home_dir}/{key}.csv", label)
        if len(tweets) > 0:
            print(f"{key}:{tweets}")
            length -= 1
    return


if __name__ == "__main__":
    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/prompts"
    system_prompt = (
    "You are an assistant to a psychiatrist. "
    "Your task is to analyze a patient's posts or dialogue and evaluate their Big Five personality traits "
    "based on linguistic, behavioral, and emotional cues. "
    "Be objective, use evidence from the text, and avoid subjective speculation. "
    "Output only a Python dictionary in the required format."
)
    user_prompt = (
    "You will receive a collection of posts written by a patient with depression, "
    "or a dialogue between this patient and a psychiatrist/psychological counselor. "
    "Your task is to evaluate the patient's Big Five personality traits based on the content. "
    "Score each of the following five dimensions on a 0–7 scale, where 0 means extremely low and 7 means extremely high. "
    "Use clues in their interests, behaviors, and language style, and avoid subjective assumptions:\n"
    "1. Openness: higher if imaginative, curious, or open to new experiences; lower if conservative or cautious.\n"
    "2. Conscientiousness: higher if organized, efficient, or self-disciplined; lower if careless, impulsive, or unstructured.\n"
    "3. Extraversion: higher if outgoing, energetic, or sociable; lower if quiet, reserved, or prefers solitude.\n"
    "4. Agreeableness: higher if kind, empathetic, or cooperative; lower if critical, antagonistic, or insensitive.\n"
    "5. Neuroticism: higher if anxious, moody, or emotionally unstable; lower if calm, emotionally resilient, or stable.\n\n"

    "If information for a dimension is missing, give a conservative low score or note the uncertainty in reasoning. "
    "Finally, output the result strictly in the following Python dictionary format:\n\n"
    "{'id': {'Openness': 0, 'Conscientiousness': 0, 'Extraversion': 0, 'Agreeableness': 0, 'Neuroticism': 0}}\n\n"

    "Note: This output format is for reference only and does not represent actual content. "
    "Below are the user ID and the collection of posts (or dialogue), each starting with the timestamp."
)
    print("yes")
    # create_bf_prompts(output_dir, system_prompt, user_prompt, max_tweets=1500, max_length=3000)


    label = "age"
    # create_judge_prompts(output_dir, system_prompt, user_prompt, label, max_length=3000)

    input_file = f"/hpc_stor03/sjtu_home/baihan.li/deprofile/prompts/{label}_judge_prompts.json"  # 替换为你的文件名
    # output_file = extract_first_n_lines(input_file, n=50)

    # qwenoutput = f"/hpc_stor03/sjtu_home/baihan.li/deprofile/qwenoutput/{label}_judge_prompts"
    # label = "work_status"
    qwenoutput = f"/hpc_stor03/sjtu_home/baihan.li/deprofile/qwenoutput/bf_prompts_part1"
    # clear_label_output("", qwenoutput, save=False, maxlength=50)
    
    # statistics_bf(max_tweets=1500)
    # print_flag_true(label, 50)
    parse_bf_results()