import pandas as pd
import os
import json
from tqdm import tqdm
import re

def construct_anchor_df():
    source_dir = "/hpc_stor03/sjtu_home/baihan.li/AIAgent/datasets/twitterSMHD/depression"
    output_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/anchor_depression.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    idx = os.listdir(source_dir)
    df = pd.DataFrame(columns=["user_id", "tweet", "date", "label"])
    for i in tqdm(idx):
        with open(os.path.join(source_dir, str(i), "anchor_tweet.json"), "r") as f:
            data = json.load(f)
            df.loc[len(df)] = [i, data["anchor_tweet"], data["anchor_tweet_date"], None]
    df.to_csv(output_path, index=False)
    
def find_time_range():
    source_dir = "/hpc_stor03/sjtu_home/baihan.li/AIAgent/datasets/twitterSMHD/depression"
    logging_file = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/anchor_depression.csv"
    csv = pd.read_csv(logging_file)
    idx = csv['user_id'].tolist()
    csv.set_index("user_id", inplace=True)
    maxdate = []
    mindate = []
    for i in tqdm(idx):
        with open(os.path.join(source_dir, str(i), "tweets.json"), "r") as f:
            data = json.load(f)
        data_list = list(data.keys())
        data_list.sort()
        mindate.append(data_list[0])
        maxdate.append(data_list[-1])
    csv['earliest_date'] = mindate
    csv['latest_date'] = maxdate
    csv.to_csv(logging_file)

def create_judge_prompts(output_dir, length=None):

    df = pd.read_csv("/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/anchor_depression.csv")
    
    if length :
        df = df.iloc[:length]

    system_prompts = f"你是一个精神科医生的助手，你需要做一个时间推测并只输出一个时间，格式为yyyy-mm-dd，即阅读用户的帖子判断用户被确诊抑郁症的时间："

    for i in range(len(df)):
        tweet = df.iloc[i]['tweet']
        anchor_date = df.iloc[i]['date']
        earliest_date = df.iloc[i]['earliest_date']

        user_prompt = f"你将收到一个抑郁症患者发的一条self-reported的英文帖子，这条帖子的发帖时间（当前时间）和收录的用户最早帖子的发帖时间（最早时间），用户在这个帖子中明确了自己被诊断为抑郁症，你需要做如下的推测：1.如果用户表示是最近被诊断为抑郁症，比如使用了now current recent之类的词汇，请你输出当前时间，2.如果用户写了明确的时间点，比如three years ago，one month ago，请你在当前时间的基础上做减法，直接在年、月、日上进行减法，然后请你判断这个是不是在最早时间之前，如果比最早时间还早，那么就输出最早时间，否则输出计算的时间，3.如果用户没有写明确的时间点，或者写了很多年以前，或者表示小时候，则输出最早时间，下面是两个例子：example1: 用户帖子：I was diagnosed depression one month ago..... 当前时间：2025-04-10，最早时间：2023-01-01, 则输出2025-03-10，因为是一个月之前被诊断; example2: 用户帖子： when I was 16, doctor said I have depression.... 当前时间：2023-04-10， 最早时间：2020-09-09，则输出：2020-09-09，判断用户被诊断为抑郁症是很久之前的事情，example3: 用户帖子：I was diagnosed depression 10 years ago, 当前时间: 2020-09-08, 最早时间：2017-01-05, 最早时间比诊断时间要晚，因此输出2017-01-05。请只输出一个格式为yyyy-mm-dd的时间字符串，请注意输出字符串的时间至少要在最早时间之后，下面是你要判断的内容，用户帖子：{tweet}, 当前时间：{anchor_date}, 最早时间：{earliest_date}"
        
        tmp = {"messages": [{"role": "system", "content": system_prompts},
                            {"role": "user", "content": user_prompt,}],}
                            
        with open(output_dir, "a", encoding="utf-8") as f:
            json.dump(tmp, f, ensure_ascii=False)
            f.write("\n")

def clean_translate_output(csv_path, llm_response_file):
    
    
    df = pd.read_csv(csv_path)
    with open(llm_response_file, "r", encoding='utf-8') as f:
        responses = f.readlines()
    for i in tqdm(range(len(responses))):
        responses[i] = responses[i].strip()
        
        responses[i] = re.sub(r"<think>.*?</think>", "", responses[i], flags=re.DOTALL).strip()
        
        responses[i] = json.loads(responses[i])["response"].replace("\n", "")
        print(responses[i])
        # break
        df.at[i, 'label'] = responses[i]
    df.to_csv(csv_path.replace('.csv', '_labeled.csv'), index=False)
    
def count_tweets_per_user():
    
    source_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/stmhd_labels/depression"
    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered"
    os.makedirs(output_dir, exist_ok=True)
    count_user = 0
    count_tweets = []
    for idx in tqdm(os.listdir(source_dir)):
        # judge if idx is can be converted to int
        try:
            int(idx)
            count_user += 1
        except ValueError:
            print(f"Skipping non-integer directory: {idx}")
            continue
        user_dir = os.path.join(source_dir, idx)
        user_df = pd.DataFrame(columns=["tweet", "date", "tweet_id", "disorder_flag"])
        with open(os.path.join(user_dir, "tweets_filtered.json"), "r") as f:
            data = json.load(f)
            for date, tweets_list in data.items():
                for tweet in tweets_list:
                    user_df.loc[len(user_df)] = [tweet['text'], date, tweet['tweet_id'], tweet["disorder_flag"]]
        user_df.to_csv(os.path.join(output_dir, f"{idx}.csv"), index=False)
        count_tweets.append(len(user_df))
    print(f"Total users: {count_user}")
    print(f"Total tweets: {sum(count_tweets)}")
    print(f"Average tweets per user: {sum(count_tweets)/count_user}")
    
def construct_csv(output_path):
    source_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered"
    idlist = list(os.listdir(source_dir))
    save_df = pd.DataFrame(columns=["user_id", "number_of_tweet", "used_tweet","gender", "age", "martial_state"])
    for idx in tqdm(idlist):
        try:
            user_df = pd.read_csv(os.path.join(source_dir, idx))
        except Exception as e1:
            print("① 标准读取失败 →", e1, idx)
            continue
        save_df.loc[len(save_df)] = [idx.replace(".csv", ""), len(user_df), min(len(user_df), 1000), None, None, None]
    save_df.to_csv(output_path, index=False)

def select_users_tweets(output_path, system_prompt, user_prompt):
    source_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd_filtered"
    source_csv = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/user_info.csv"
    idlist = pd.read_csv(source_csv)['user_id'].tolist()
    save_df = pd.DataFrame(columns=["user_id", "number_of_tweet", "used_tweet","gender", "age", "martial_state"])
    for idx in tqdm(idlist):
        user_df = pd.read_csv(os.path.join(source_dir, str(idx)+".csv"))
        save_df.loc[len(save_df)] = [idx, len(user_df), min(len(user_df), 1000), None, None, None]
        
        if len(user_df) <= 1000:
            tweets = user_df['tweet'].tolist()
        else:
            # select disorder flag = True tweets first
            tweets = user_df[user_df['disorder_flag'] == True]['tweet'].tolist()
            if len(tweets) < 1000:
                remaining = 1000 - len(tweets)
                # select randomly from disorder_flag = False tweets
                tweets += user_df[user_df['disorder_flag'] == False]['tweet'].sample(n=remaining, random_state=42).tolist()
            else:
                tweets = tweets[:1000]
        for tweet in tweets:
            tmp = {"messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt+tweet},],}
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(tmp, f, ensure_ascii=False)
                f.write("\n")
                
def split_json(input_file, parts=3):
    # 按行读取
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    chunk_size = (total + parts - 1) // parts  # 向上取整

    output_prefix = input_file.replace(".json", "")
    for i in range(parts):
        start = i * chunk_size
        end = start + chunk_size
        chunk = lines[start:end]

        if not chunk:
            break
    
        output_file = f"{output_prefix}_part{i+1}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(chunk)

        print(f"已生成: {output_file} (包含 {len(chunk)} 条记录)")

def clear_label_output(label_name):
    qwenoutput = f"/hpc_stor03/sjtu_home/baihan.li/deprofile/qwenoutput/judge_{label_name}_all"
    with open(qwenoutput, "r", encoding="utf-8") as f:
        responses = f.readlines()
    df = pd.read_csv(f"/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/user_info.csv")
    for i in tqdm(range(len(df))):
        
        this_response = responses[0:df.at[i, 'used_tweet']]
        responses = responses[df.at[i, 'used_tweet']:]
        
        for j in range(len(this_response)):
            this_response[j] = this_response[j].strip()
            this_response[j] = re.sub(r"<think>.*?</think>", "", this_response[j], flags=re.DOTALL).strip()
            this_response[j] = json.loads(this_response[j])["response"].replace("\n", "")
            this_response[j] = this_response[j].strip()
            this_response[j] = re.sub(r"['\"]", "", this_response[j])
        # only for test: print all count results
        # print(f"User {df.at[i, 'user_id']} - Label counts: ")
        # print(f"Female: {this_response.count('F')}")
        # print(f"Male: {this_response.count('M')}")
        # print(f"Unknown: {this_response.count('Unknown')}")
        # delete "Unknown" responses
        this_response = [resp for resp in this_response if resp != "Unknown"]
        # majority vote
        if len(this_response) == 0:
            final_label = "Unknown"
        else:
            final_label = max(set(this_response), key=this_response.count)
            
        
        # break
        df.at[i, label_name] = final_label
    df.to_csv(f"/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/user_info.csv", index=False)
    
def compare():
    with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/stmhd_profile.json", "r", encoding="utf-8") as f:
        profiles = f.readlines()
    df = pd.read_csv("/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/user_info.csv")
    
    for line in profiles:
        gender = json.loads(line)["gender"]
        line = json.loads(line)
        gender_me = df[df['user_id'] == int(line['id'])]['gender'].values[0]
        print(line["id"], f"bingrui: {gender}, me: {gender_me}")
        
if __name__ == "__main__":
    # find_time_range()
    # construct_anchor_df()
    # count_tweets_per_user()

    output_dir = "/hpc_stor03/sjtu_home/baihan.li/deprofile/prompts/diagnosed_data_predict.json"
    # create_judge_prompts(output_dir)
    
    csv_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/anchor_depression.csv"
    llm_response_file = "/hpc_stor03/sjtu_home/baihan.li/deprofile/qwenoutput/diagnosed_data_predict"
    # clean_translate_output(csv_path, llm_response_file)
    
    
    # construct_csv(output_path="/hpc_stor03/sjtu_home/baihan.li/deprofile/data/smhd/user_info.csv")
    output_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/prompts/judge_age.json"
    system_prompt = f"你是一个精神科医生的助手，你需要阅读抑郁症的帖子并根据要求打年龄标签,并仅输出一个结果的python字符串。"
    user_prompt = "/no_thinking你将收到一个用户的一条帖子：请判断这条帖子能否确定用户的年龄按以下规则输出一个 python的字符串,请在 ['0-17','18-25','26-35','36-50','50+','Unknown']中选择，这些年龄段分别对应未成年人，青年大学生，初入职场的青年人，中年人，老年人，和内容中无任何可以判断年龄的信息。请只在出现表标志性某个年龄段的标志性事件或者直接提到自己的年龄时给出，否则请输出unknown。下面是几个例子：example1：用户帖子：Today is my 22nd birthday... 输出：18-25，用户明确提到了自己的年龄、因此可以直接判断用户的年龄，example2：用户帖子：Last year I retired from Amazon—felt like... 输出：50+, 用户明确提到了退休这一绝大部分发生在50岁以上人员群体的，example3:I’m obsessed with Taylor—her songs are on repeat every single day. 输出：Unknown，尽管歌手泰勒的粉丝群体是年轻人的概率更高，但其覆盖的年龄阶段较为宽泛，因此无法确定用户的年龄。请只输出一个字符串，下面是用户帖子："
    # select_users_tweets(output_path, system_prompt, user_prompt)
    

    output_path = "/hpc_stor03/sjtu_home/baihan.li/deprofile/prompts/all_life_event_prompts.json"

    split_json(output_path,4)
    
    # clear_label_output("gender")
    
    # compare()

    