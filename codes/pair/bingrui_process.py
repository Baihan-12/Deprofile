import pickle
import os 
import sys
import re 
import csv
import json 
import shutil
import tiktoken
# from collections import defaultdict, Counter 
# from tqdm import tqdm
# import random
import pickle
import pandas as pd
import numpy as np
# from career.career import classify_user
# from career.conclude_tweets import conclude_tweets
# from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import umap
from scipy.stats import gaussian_kde
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# api_key = "sk-ccKY17WB0lJpAyzMD822075f5a8c44A2Ab239fBaE1291e14"
# api_base = "https://api.xi-ai.cn/v1"
# client = OpenAI(api_key = api_key, base_url=api_base)

diseases = ['adhd', 'anxiety', 'bipolar', 'depression', 'neg', 'ocd', 'ptsd']

life_events_list = [
    'Career', 'Death', 'Education', 'Financial', 'Health',
    'Identity', 'Legal', 'Lifestyle_Change', 'New_Birth_in_Family', 
    'Relationships_Changes', 'Relocation', 'Societal'
]

symptoms_list = [
    "Anxious_Mood","Autonomic_symptoms","Cardiovascular_symptoms","Catatonic_behavior",
    "Decreased_energy_tiredness_fatigue","Depressed_Mood",
    "Gastrointestinal_symptoms","Genitourinary_symptoms","Hyperactivity_agitation",
    "Impulsivity","Inattention","Indecisiveness","Respiratory_symptoms","Suicidal_ideas",
    "Worthlessness_and_guilty","avoidance_of_stimuli",
    "compensatory_behaviors_to_prevent_weight_gain","compulsions",
    "diminished_emotional_expression","do_things_easily_get_painful_consequences","drastical_shift_in_mood_and_energy",
    "fear_about_social_situations","fear_of_gaining_weight","fears_of_being_negatively_evaluated","flight_of_ideas",
    "intrusion_symptoms","loss_of_interest_or_motivation",
    "more_talktive","obsession","panic_fear","pessimism","poor_memory",
    "sleep_disturbance","somatic_muscle","somatic_symptoms_others","somatic_symptoms_sensory",
    "weight_and_appetite_change","Anger_Irritability"
]
career_domain_list = [
    "Creative Arts and Media", "Business and Finance", "Technology and Engineering", "Healthcare and Social Services",
    "Education and Research", "Legal and Public Policy", "Transportation and Logistics", "Manufacturing and Construction", 
    "Hospitality and Tourism"
]

work_status_list = ["employed", "unemployed", "retired", "student"]

# 职业和工作状态关键词库
career_keywords = {
    "Creative Arts and Media": ["Art", "Artist", "Design", "Graphic Design", "Illustration", "Animation", "Film", "Photography", "Actor", "Music", "Composer", "Director", "Cinematography", "Performing Arts", "Fashion", "Model", "Stylist", "Video Production", "Copywriter", "Journalist", "Photographer", "Film Production", "Theatre", "Singer"],
    "Business and Finance": ["Business", "Entrepreneur", "Finance", "Accounting", "Investment", "Sales", "Marketing", "Advertising", "Analyst", "Strategy", "Operations", "Project Manager", "Business Development", "Corporate", "HR", "Retail", "Branding", "Consumer", "Financial Planning", "Budgeting", "Trading", "Tax", "Commercial, Banking", "Business Analyst", "Consultant", "Real Estate", "Negotiation"],
    "Technology and Engineering": ["Engineer", "Mechanical", "Civil Engineer", "Electrical Engineer", "Software Developer", "IT", "Data Scientist", "Cloud Computing", "Artificial Intelligence", "Cybersecurity", "Coding", "Programming", "Computer Science", "IT Support", "Network Engineer", "Web Development", "Database", "Network Security", "Data Analytics", "Robotics", "Software Architecture", "Engineering", "DevOps", "IT Management", "Software Engineer", "Algorithm", "Python", "Java", "C++"],
    "Healthcare and Social Services": ["Healthcare", "Medical", "Doctor", "Nurse", "Physician", "Therapist", "Social Worker", "Caregiver", "Public Health", "Psychologist", "Counselor", "Mental Health", "Pharmacist", "Physician Assistant", "Nurse Practitioner", "Public Safety", "Emergency", "Hospital", "Health Services", "Medical Research", "Geriatrics", "Pediatrics", "Surgery", "Hospital Administration", "Nursing", "Health Policy", "Rehabilitation", "Healthcare Assistant", "Psychotherapy"],
    "Education and Research": ["Teacher", "Educator", "Instructor", "Professor", "Academic", "Researcher", "Research Assistant", "Science", "STEM", "Classroom", "School", "Curriculum", "Training", "Learning", "Teaching", "Academic Research", "Laboratory", "Biologist", "Chemist", "Physicist", " Educational Technology", " Educational Psychology", "Study", "PhD", "University", "Tutor", "Pedagogy", "Higher Education", "Teaching Assistant", "Science Research", "Study Abroad"],
    "Legal and Public Policy": ["Lawyer", "Attorney", "Legal", "Law", "Legal Counsel", "Paralegal", "Court", "Litigation", "Prosecution", "Defense", "Judge", "Legal Assistant", "Policy", "Government", "Public Policy", "Compliance", "Tax Law", "Corporate Law", "Intellectual Property", "Criminal Justice", "Regulatory", "Public Defender", "Government Relations", "Public Safety", "Crime", "Police", "Legislation", "Human Rights", "Ethics"],
    "Transportation and Logistics": ["Logistics", "Transport", "Shipping", "Freight", "Warehouse", "Distribution", "Driver", "Pilot", "Air Traffic Controller", "Delivery", "Cargo", "Fleet", "Route", "Shipping Management", "Supply Chain", "Logistics Coordinator", "Truck Driver", "Transportation Management", "Customs", "Cargo", "Freight Forwarding", "Shipping Company", "Distribution Center", "Trucking", "Aviation", "Rail Transport", "Dispatch", "Shipping Services"],
    "Manufacturing and Construction": ["Construction", "Building", "Carpenter", "Architect", "Civil Engineer", "Mechanical Engineer", "Welder", "Technician", "Laborer", "Factory", "Assembly", "Site Manager", "Project Manager", "Architecture", "Blueprints", "Design", "Quality Control", "Manufacturing", "Production", "Industrial", "Heavy Machinery", "Welding", "Manufacturing Plant", "Contractor", "Structural Engineering", "Construction Manager", "Safety"],
    "Hospitality and Tourism": ["Hotel", "Resort", "Travel", "Tourism", "Tour Guide", "Vacation", "Restaurant", "Chef", "Waiter", "Bartender", "Concierge", "Event Planning", "Catering", "Hospitality", "Tourist", "Booking", "Leisure", "Entertainment", "Tourism Industry", "Travel Agency", "Event Coordinator", "Resort Management", "Cruise", "Flight", "Tourist Destination", "Hotel Management", "Tourism Consultant", "Hospitality Management"]
}

# 工作状态相关关键词
work_status_keywords = {
    # 就业相关状态
    "employed": [
        "employed", "working", "job", "employee", "full-time", "part-time", "working full-time", "working part-time", 
        "on payroll", "in office", "hired", "job holder", "active employment", "working at", "currently employed", 
        "workplace", "office", "officially employed", "working from home"
    ],
    
    # 失业相关状态
    "unemployed": [
        "unemployed", "jobless", "seeking job", "looking for work", "not working", "out of work", "looking for a job", 
        "unemployment", "job seeker", "currently unemployed", "seeking new opportunities", "job hunting", "unemployed at the moment",
        "unhired", "in between jobs", "without employment", "currently not working", "displaced worker", "jobless at the moment"
    ],
    
    # 退休相关状态
    "retired": [
        "retired", "retirement", "pension", "not working", "retired from", "pensioned", "retiring", "post-retirement", 
        "enjoying retirement", "retirement life", "life after retirement", "semi-retired", "retirement benefits", "retired early", 
        "not employed", "retired from work", "having retired"
    ],
    
    "student": [
        "student", "studying", "in school", "currently a student", "school", "university", "high school", "college student", 
        "pursuing degree", "currently enrolled", "in class", "on campus", "full-time student", "part-time student", 
        "taking courses", "academic break", "research student"
    ]
}

def get_direct_subfolders(folder_path):
    # 获取文件夹下所有直接子文件夹
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subfolders

def read_json(path):
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def read_npy(path):
    data = np.load(path)
    return data

def write_json(path, data):
    with open (path, mode="w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print("Successfully  write to json!!!")
    
def copy(src_folder, dst_folder):
    shutil.copytree(src_folder, dst_folder)
    print(f'{src_folder} to {dst_folder}!')

def copy_folder(src, dest):
    # 检查源文件夹是否存在
    if not os.path.exists(src):
        print(f"源文件夹 {src} 不存在！")
        return
    
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(dest):
        os.makedirs(dest)
        print(f"目标文件夹 {dest} 已创建！")
    
    # 执行文件夹复制
    try:
        shutil.copytree(src, os.path.join(dest, os.path.basename(src)))
        print(f"文件夹 {src} 成功复制到 {dest}")
    except Exception as e:
        print(f"复制失败：{e}")

def write_to_csv(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        for row in data:
            csv_writer.writerow(row)

def append_to_json(path, new_data):
    data = read_json(path)
    data.append(new_data)
    write_json(path, data)

def add_quotes(sentence):
    sentence = str(sentence)
    # 如果句子已经被引号包裹，则直接返回
    if (sentence.startswith('"') and sentence.endswith('"')) or (sentence.startswith(''') and sentence.endswith(''')):
        return sentence
    # 否则，添加引号
    return f'"{sentence}"'

def convert_hashtag(tag):
    tag = tag.strip("#")
    tag = re.findall(r'[A-Z]+[a-z]*|\d+', tag)
    tag = "".join(tag)
    tag = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', tag)
    tag = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
    return tag

def clean_tweet(text, do_convert_hashtag=True, remove_emoji=True):
    if do_convert_hashtag:
        hashtags = re.findall(r'#\w+', text)
        for hashtag in hashtags:
            converted_hashtag = convert_hashtag(hashtag)
            text = text.replace(hashtag, converted_hashtag)
    text = re.sub(r"[\u200b-\u200d]", "", text)
    text = re.sub(r"(\\u200b|\\u200c|\\u200d)", "", text)
    try:
        URL_REGEX = re.compile(
            r'(?i)http[s]?://(?:[a-zA-Z]|[0-9]|[#$%*-;=?&@~.&+]|[!*,])+',
            re.IGNORECASE)
        text = re.sub(URL_REGEX, "", text)
    except:
        zh_puncts1 = "，；、。！？（）《》【】"
        URL_REGEX = re.compile(
            r'(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>' + zh_puncts1 + ']+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’' + zh_puncts1 + ']))',
            re.IGNORECASE)
        text = re.sub(URL_REGEX, "", text)
    if remove_emoji:
        emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
    return text.strip()

def load_variables_by_name(prompt, variable_dict):
        placeholders = re.findall(r'!{(\w+)}!', prompt)
        cnt = 0
        for placeholder in placeholders:
            prompt = prompt.replace(f'!{{{placeholder}}}!', str(variable_dict[placeholder]))
            cnt += 1
        prompt = prompt.strip()
        return prompt


# record the id of the users we choose
# nb of tweets before 2021 >500
# uid/{disease}.json中的顺序和pkl文件中的顺序是一致的
def usr_id(disease):
    data_path = f"./TwitterSMHD/sel_data_{disease}.pkl"
    write_path = f"./TwitterSMHD/dat/uid/{disease}.json"
    data = read_pkl(data_path)
    uid_list = list(data.keys())
    write_json(write_path, uid_list)

# gather the original tweets(>2021) of choosen users
def make_orig_tweet(disease):
    id_list_path = f"./TwitterSMHD/dat/uid/{disease}.json"
    id_list = read_json(id_list_path)
    for id in id_list:
        output_list = []
        if not os.path.exists(f"./TwitterSMHD/dat/original/{disease}/{id}"):
            os.makedirs(f"./TwitterSMHD/dat/original/{disease}/{id}")
        write_path = f"./TwitterSMHD/dat/original/{disease}/{id}/tweets.json"
        data_path = f"./TwitterSMHD/{disease}/{id}/tweets.json"
        tweets = read_json(data_path)
        for timestamp in tweets:
            if int(timestamp[:4]) < 2021:
                print(timestamp)
                output_list.append(tweets[timestamp])
        write_json(write_path, output_list)
        
# disease = ['adhd', 'anxiety', 'depression', 'neg', 'bipolar', 'ocd', 'ptsd']
# mode = ['le', 'symp']
# split = ['disease', 'user']
def process_input(disease, mode, split):
    print(disease)
    if mode == 'le':
        header = [
            'disease', 'sentence', 'Career', 'Death', 'Education', 'Financial', 
            'Health', 'Identity', 'Legal', 'Lifestyle_Change', 'New_Birth_in_Family', 
            'Relationships_Changes', 'Relocation', 'Societal', 'uncertain'
        ]
        pkl_path = f"./TwitterSMHD/sel_data_{disease}.pkl"
        data = read_pkl(pkl_path)
        content = []
        if split == 'disease':
            for uid in data:
                for text in data[uid]:
                    sentence = clean_tweet(text)
                    sentence = add_quotes(sentence)
                    content.append(sentence)
            if not os.path.exists(f'./Life_Events/LE_classifier/input/le/{disease}/'):
                os.makedirs(f'./Life_Events/LE_classifier/input/le/{disease}/')
            write_path = f'./Life_Events/LE_classifier/input/le/{disease}/test.csv'
            with open(write_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for sentence in content:
                    row = ['none', sentence] + [0] * (len(header) - 2)
                    writer.writerow(row)
        elif split == 'user':
            for uid in data:
                content = []
                if not os.path.exists(f'./Life_Events/LE_classifier/input/le/user/{disease}/{uid}/'):
                    os.makedirs(f'./Life_Events/LE_classifier/input/le/user/{disease}/{uid}/')
                for text in data[uid]:
                    sentence = clean_tweet(text)
                    sentence = add_quotes(sentence)
                    content.append(sentence)
                write_path = f'./Life_Events/LE_classifier/input/le/user/{disease}/{uid}/test.csv'
                with open(write_path, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                    for sentence in content:
                        row = ['none', sentence] + [0] * (len(header) - 2)
                        writer.writerow(row)   
    elif mode == 'symp':
        header = [
            "conversation_id","tweet_id","sentence_id","disease","sentence",
            "Anxious_Mood","Autonomic_symptoms","Cardiovascular_symptoms","Catatonic_behavior",
            "Decreased_energy_tiredness_fatigue","Depressed_Mood",
            "Gastrointestinal_symptoms","Genitourinary_symptoms","Hyperactivity_agitation",
            "Impulsivity","Inattention","Indecisiveness","Respiratory_symptoms","Suicidal_ideas",
            "Worthlessness_and_guilty","avoidance_of_stimuli",
            "compensatory_behaviors_to_prevent_weight_gain","compulsions",
            "diminished_emotional_expression","do_things_easily_get_painful_consequences","drastical_shift_in_mood_and_energy",
            "fear_about_social_situations","fear_of_gaining_weight","fears_of_being_negatively_evaluated","flight_of_ideas",
            "intrusion_symptoms","loss_of_interest_or_motivation",
            "more_talktive","obsession","panic_fear","pessimism","poor_memory",
            "sleep_disturbance","somatic_muscle","somatic_symptoms_others","somatic_symptoms_sensory",
            "weight_and_appetite_change","Anger_Irritability"
        ]
        id_list_path = f"./TwitterSMHD/dat/uid/{disease}.json"
        id_list = read_json(id_list_path)
        if split == 'disease':
            content = []
            tweet_id_list = []
            conv_id_list = []
            sent_id_list = []
            sentence_id = 0
            if not os.path.exists(f"PsySym/data/tweets/{disease}/"):
                os.makedirs(f"PsySym/data/tweets/{disease}/")
            write_path = f"PsySym/data/tweets/{disease}/test_new.csv"
            for id in id_list:
                data_path = f"./TwitterSMHD/dat/original/{disease}/{id}/tweets.json"
                data = read_json(data_path)
                for date_tweet in data:
                    for tweet in date_tweet:
                        tweet_id = tweet['tweet_id']
                        conversation_id = tweet['conversation_id']
                        text = tweet['text']
                        sentence = add_quotes(clean_tweet(text))
                        tweet_id_list.append(tweet_id)
                        conv_id_list.append(conversation_id)
                        sent_id_list.append(sentence_id)
                        content.append(sentence)
                        sentence_id += 1
            with open(write_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for i in range(len(content)):
                    row = [conv_id_list[i], tweet_id_list[i], sent_id_list[i], disease, content[i]] + [0] * (len(header) - 5)
                    writer.writerow(row)
            print("Sucessfully write to csv file!!!")   
        elif split == 'user':
            for id in id_list:
                content = []
                tweet_id_list = []
                conv_id_list = []
                sent_id_list = []
                sentence_id = 0
                data_path = f"./TwitterSMHD/dat/original/{disease}/{id}/tweets.json"
                data = read_json(data_path)
                if not os.path.exists(f"PsySym/data/tweets/user/{disease}/{id}/"):
                    os.makedirs(f"PsySym/data/tweets/user/{disease}/{id}/")
                write_path = f"PsySym/data/tweets/user/{disease}/{id}/test.csv"
                for date_tweet in data:
                    for tweet in date_tweet:
                        tweet_id = tweet['tweet_id']
                        conversation_id = tweet['conversation_id']
                        text = tweet['text']
                        sentence = add_quotes(clean_tweet(text))
                        tweet_id_list.append(tweet_id)
                        conv_id_list.append(conversation_id)
                        sent_id_list.append(sentence_id)
                        content.append(sentence)
                        sentence_id += 1
                with open(write_path, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                    for i in range(len(content)):
                        row = [conv_id_list[i], tweet_id_list[i], sent_id_list[i], disease, content[i]] + [0] * (len(header) - 5)
                        writer.writerow(row)
                print("Sucessfully write to csv file!!!")   

def slice_input(disease, mode):
    if mode == 'le':
        input_path = f'Life_Events/LE_classifier/input/le/{disease}/test.csv'
        output_folder = f'Life_Events/LE_classifier/input/slice/{disease}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        data = pd.read_csv(input_path)
        chunk_size = 5000000
        for i, start_row in enumerate(range(0, len(data), chunk_size)):
            # 切片数据
            chunk = data.iloc[start_row:start_row + chunk_size]
            # 保存文件
            output_folder = f'Life_Events/LE_classifier/input/slice/{disease}/part_{i+1}'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            chunk_file_path = os.path.join(output_folder, 'test.csv')
            chunk.to_csv(chunk_file_path, index=False)
            print(f"Part {i+1} saved to {chunk_file_path}")
    elif mode == 'symp':
        input_path = f'PsySym/data/tweets/{disease}/test.csv'
        output_folder = f'PsySym/input/slice/{disease}'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        data = pd.read_csv(input_path)
        chunk_size = 5000000
        for i, start_row in enumerate(range(0, len(data), chunk_size)):
            # 切片数据
            chunk = data.iloc[start_row:start_row + chunk_size]
            # 保存文件
            output_folder = f'PsySym/input/slice/{disease}/part_{i+1}'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            chunk_file_path = os.path.join(output_folder, 'test.csv')
            chunk.to_csv(chunk_file_path, index=False)
            print(f"Part {i+1} saved to {chunk_file_path}")
def decode(input_path, output_path):
    test=np.load(input_path) 
    df = pd.DataFrame(test)
    df.to_csv(output_path, index=False, float_format='%.10f')  # 保留10位小数精度
    print(f'Decode {input_path} successfully to {output_path}!!!')
def read_pred(path):
    data = np.load(path)
    print(data.shape)
    print(data[:20])
def process_pred(res_path, tweet_path, seuil):
    data = np.load(res_path)
    # rows, cols = np.where(data > seuil)
    tweet = pd.read_csv(tweet_path)
    write_path = 'TweetMind/symp/ocd/res.csv'
    header = ['id', 'text', 'Anxious_Mood','Autonomic_symptoms','Cardiovascular_symptoms',
              'Catatonic_behavior','Decreased_energy_tiredness_fatigue','Depressed_Mood',
              'Gastrointestinal_symptoms','Genitourinary_symptoms','Hyperactivity_agitation',
              'Impulsivity','Inattention','Indecisiveness','Respiratory_symptoms','Suicidal_ideas',
              'Worthlessness_and_guilty','avoidance_of_stimuli','compensatory_behaviors_to_prevent_weight_gain',
              'compulsions','diminished_emotional_expression','do_things_easily_get_painful_consequences',
              'drastical_shift_in_mood_and_energy','fear_about_social_situations','fear_of_gaining_weight',
              'fears_of_being_negatively_evaluated','flight_of_ideas','intrusion_symptoms','loss_of_interest_or_motivation',
              'more_talktive','obsession','panic_fear','pessimism','poor_memory','sleep_disturbance','somatic_muscle','somatic_symptoms_others',
              'somatic_symptoms_sensory','weight_and_appetite_change','Anger_Irritability'
        
    ]
    rows_with_condition = np.unique(np.where(data > 0.5)[0])
    filtered_rows = data[rows_with_condition]
    text = []
    h_disease = tweet.columns.tolist()[4:]
    for row in rows_with_condition:
        sentence = tweet.loc[row, 'sentence']
        text.append(sentence)
    with open(write_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ['row', 'sentence'] + h_disease
        writer.writerow(header)
        for i in range(len(text)):
            row = [rows_with_condition[i], text[i], filtered_rows[i]]
            writer.writerow(row)    
def dataset(disease):
    id_list_path = f'TweetMind/data/uid/{disease}.json'
    id_list = read_json(id_list_path)
    for id in id_list:
        # source_file1 = f'TwitterSMHD/{disease}/{id}/anchor_tweet.json'
        source_file2 = f'TwitterSMHD/{disease}/{id}/user.json'
        destination_folder = f'TweetMind/data/original/{disease}/{id}/'
        # shutil.copy(source_file1, destination_folder)
        shutil.copy(source_file2, destination_folder)
        
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

    matched_sentences = []  # 用于存储匹配到的完整句子
    for pattern in age_patterns:
        matches = re.findall(pattern, text)
        if matches and text not in matched_sentences:
            matched_sentences.append(text)  # 如果匹配到，就返回完整的原文
    
    return matched_sentences if matched_sentences else None                      

def age_tweet(disease):
    folder_path = f'TweetMind/data/original/{disease}'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for subfolder in subfolders:
        input_path = f'{folder_path}/{subfolder}/tweets.json'
        write_path = f'{folder_path}/{subfolder}/age_tweets.json'
        tweets = read_json(input_path)
        output = []
        for date_tweet in tweets:
            for tweet in date_tweet:
                text = tweet['text']
                age_tweet = extract_age(text)
                if age_tweet:
                    output.append(tweet)
        write_json(write_path, output)
        
def copy_age_tweets(disease):
    input_folder = f'/hpc_stor03/sjtu_home/bingrui.jin/codes/TweetMind/data/original/{disease}'
    subfolders = get_direct_subfolders(input_folder)
    for udir in subfolders:
        source_file = f'{udir}/age_tweets.json'
        prefix_path = f'/hpc_stor03/sjtu_home/bingrui.jin/codes/TweetMind/data/original/{disease}'
        if udir.startswith(prefix_path):
            uid = udir[len(prefix_path):].lstrip("/") 
        destination_folder = f'/hpc_stor03/sjtu_home/bingrui.jin/codes/TweetMind/copy/{disease}/{uid}'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)        
        shutil.copy(source_file, destination_folder)
        print(f"{source_file} 已复制到 {destination_folder}")
        
def reconstruct(disease):
    id_path = f'data\\uid\\{disease}.json'
    id_list = read_json(id_path)
    for id in id_list:
        output_tweets = []
        output_age_tweets = []
        write_folder = f'data\\filter_tweets_num\\{disease}\\{id}'
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)            
        write_tweets_path = f'{write_folder}\\tweets.json'
        write_at_path = f'{write_folder}\\age_tweets.json'
        tweet_path = f'data\\org\\{disease}\\{id}\\tweets.json'
        tweets = read_json(tweet_path)
        for timestamp in tweets:
            if int(timestamp[:4]) < 2021:
                date_list = []
                for tweet in tweets[timestamp]:
                    date_list.append(tweet)
                    text = tweet['text']
                    age_t = extract_age(text)
                    if age_t:
                        output_age_tweets.append(tweet)
                output_tweets.append(date_list)
        write_json(write_tweets_path, output_tweets)
        write_json(write_at_path, output_age_tweets)

def extract_marital_status(text):
    marriage_patterns = {
        "married": r"(?i)\b(married|marry|spouse|wife|husband|wedding|partner|tied the knot|marital|marriage|my better half)\b",
        "single": r"(?i)\b(single|unmarried|bachelor|spinster|no boyfriend|no girlfriend|flying solo|living alone|not seeing anyone)\b",
        "divorced": r"(?i)\b(divorced|divorce|ex-husband|exhusband|ex-wife|exwife|separated|divorcee|split up|ended our marriage|custody battle|amicable separation)\b",
        "widowed": r"(?i)\b(widowed|widower|widow|lost my wife|lost my husband|late spouse|passed away|grieving my partner)\b"
    }
    for status, pattern in marriage_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return text
    return None

def marital_status_tweet(disease):
    folder_path = f'data\\filter_tweets_num\\{disease}'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for subfolder in subfolders:
        input_path = f'{folder_path}/{subfolder}/tweets.json'
        write_path = f'{folder_path}/{subfolder}/marital_status_tweets.json'
        tweets = read_json(input_path)
        output = []
        for date_tweet in tweets:
            for tweet in date_tweet:
                text = tweet['text']
                age_tweet = extract_marital_status(text)
                if age_tweet:
                    output.append(tweet)
        write_json(write_path, output)
def extract_work_status(text):
    work_patterns = {
        "Employed": r"(?i)\b(working|employed|full-time|part-time|freelancer|consultant|self-employed|entrepreneur|running a business|active in the workforce|intern|engaged in work|on a job|work|job|office|hired|employment|workplace)\b",
        "Unemployed": r"(?i)\b(unemployed|jobless|looking for a job|seeking employment|out of work|laid off|fired|between jobs|job hunting|without a job|not working|unemployed|unemployment|unhired|job seeker|without employment|displaced worker)\b",
        "Retired": r"(?i)\b(retired|pensioner|retirement|no longer working|former employee|post-career|pension|senior citizen|retire|retiring|post-retirement)\b",
        "Student": r"(?i)\b(student|studying|university|school|college|pursuing degree|enrolled|in class|on campus|taking courses|academic break)\b"
    }
    for status, pattern in work_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return text
    return None

def work_status_tweet(disease):
    folder_path = f'data\\filter_tweets_num\\{disease}'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for subfolder in subfolders:
        input_path = f'{folder_path}/{subfolder}/tweets.json'
        write_path = f'{folder_path}/{subfolder}/profile/work_status_tweets.json'
        tweets = read_json(input_path)
        output = []
        for date_tweet in tweets:
            for tweet in date_tweet:
                text = tweet['text']
                age_tweet = extract_work_status(text)
                if age_tweet:
                    output.append(tweet)
        write_json(write_path, output)
def process_gender(disease, ind):
    res_path = f'gender\\output\\{disease}\\part_1\\output.csv'
    df = pd.read_csv(res_path)
    male_val, unknown_val, female_val = df.iloc[ind]
    if (int(male_val) == 1) and (int(unknown_val) == 0) and (int(female_val) == 0):
        gender = "male"
    elif (int(male_val) == 0) and (int(unknown_val) == 0) and (int(female_val) == 1):
        gender = "female"
    else:
        gender = False
    return gender
def construct_profile(disease):
    print(disease)
    id_path = f'data\\uid\\{disease}.json'
    id_list = read_json(id_path)  
    for id in id_list:
        print('-----------------------------------------')
        print(id)
        ind = id_list.index(id)
        user_folder = f'data\\filter_tweets_num\\{disease}\\{id}'
        age_path = f'{user_folder}\\age_res.json'
        ms_path = f'{user_folder}\\ms_res.json'
        ws_path = f'{user_folder}\\ws_res.json'
        des_path = f'data\\org\\{disease}\\{id}\\user.json'
        gender_path = f'{user_folder}\\gender_res.json'
        write_path = f'{user_folder}\\profile.json'
        profile_dict = dict()
        age_data = read_json(age_path)
        ms_data = read_json(ms_path)
        ws_data = read_json(ws_path)
        des_data = read_json(des_path)
        gender_data = read_json(gender_path)
        profile_dict["id"] = id
        if age_data != []:
            age_response = age_data['response']
            if age_response:
                profile_dict['age'] = int(age_response['age'])
            else:
                profile_dict['age'] = False
        else:
            profile_dict['age'] = False
        gender_model = gender_data['model']
        gender_keywords = gender_data['keywords']['gender']
        if gender_keywords:
            profile_dict['gender'] = gender_keywords.lower()
        else:
            profile_dict['gender'] = gender_model
        if ms_data != []:
            ms_response = ms_data['response']
            if ms_response:
                profile_dict['marital_status'] = ms_response['marital_status']
            else:
                profile_dict['marital_status'] = False      
        else:
            profile_dict['marital_status'] = False
        if ws_data != []:
            ws_response = ws_data['response']
            if ws_response:
                profile_dict['work_status'] = ws_response['work_status']
            else:
                profile_dict['work_status'] = False
        else:
            profile_dict['work_status'] = False
        self_des = des_data["description"]      
        profile_dict['self_description'] = self_des
        profile_dict['life_events'] = ""
        profile_dict['symptoms'] = ""
        write_json(write_path, profile_dict)
        

def extract_gender(tweet_num, tweet, profile):
    male_keywords_tweet = [r'\bmy wife\b', r'\bmy girlfriend\b', r'\bmy gf\b', r'\bmy lady\b', r'\bmy fiancée\b', r'\bI married her\b']
    female_keywords_tweet = [r'\bmy husband\b', r'\bmy boyfriend\b', r'\bmy bf\b', r'\bmy fiancé\b', r'\bI married him\b']
    male_keywords_profile = [r'\bhe\b', r'\bhim\b', r'\bhis\b', r'\bman\b', r'\bboy\b', r'\bguy\b', r'\bmen\b', r'\bhusband\b', r'\bdad\b', r'\bfather\b', r'\bgentlemen\b', r'\bbachelor\b', r'\bbrother\b', r'\bking\b', r'\bprince\b', r'\bsir\b', r'\bmonsieux\b']
    female_keywords_profile = [r'\bshe\b', r'\bher\b', r'\bgirl\b', r'\bwoman\b', r'\blady\b', r'\bfemale\b', r'\bwife\b', r'\bmom\b', r'\bmother\b', r'\bmiss\b', r'\bmadam\b', r'\bmamm\b', r'\bprincess\b', r'\bqueen\b', r'\bfemale\b', r'\bsister\b']
    tweet_lower = tweet.lower()
    profile_lower = profile.lower()
    ratio = 1
    if tweet_num >= 5:
        ratio = 0.5
    male_count = sum(len(re.findall(keyword, tweet_lower)) for keyword in male_keywords_tweet) * ratio + sum(len(re.findall(keyword, profile_lower)) for keyword in male_keywords_profile)
    female_count = sum(len(re.findall(keyword, tweet_lower)) for keyword in female_keywords_tweet) * ratio + sum(len(re.findall(keyword, profile_lower)) for keyword in female_keywords_profile)
    
    # 根据次数确定性别
    if male_count > female_count:
        gender = "Male"
    elif female_count > male_count:
        gender = "Female"
    else:
        gender = False
    
    return {
        "description": profile,
        "male_count": male_count,
        "female_count": female_count,
        "gender": gender
    }

def clear_file(disease):
    print(disease)
    id_path = f'data\\uid\\{disease}.json'
    id_list = read_json(id_path)
    for id in id_list:
        print(f'-------------------------------{id}-----------------------------------------')
        udir = f'data\\filter_tweets_num\\{disease}\\{id}'
        at_path = f'{udir}\\age_tweets.json'
        ms_path = f'{udir}\\marital_status_tweets.json'
        ws_path = f'{udir}\\work_status_tweets.json'
        os.remove(at_path)
        os.remove(ms_path)
        os.remove(ws_path)
def remove_profile(disease):
    print(f'------------------------------------{disease}-----------------------------------------------')
    id_path = f'data\\uid\\{disease}.json'
    id_list = read_json(id_path)  
    for id in id_list:
        print(f'-----------------------{id}------------------------')
        profile_folder = f'data\\filter_tweets_num\\{disease}\\{id}\\profile'
        if not os.path.exists(profile_folder):
            os.makedirs(profile_folder)
        age_old = f'data\\filter_tweets_num\\{disease}\\{id}\\age_res.json'
        gender_old = f'data\\filter_tweets_num\\{disease}\\{id}\\gender_res.json'
        ms_old = f'data\\filter_tweets_num\\{disease}\\{id}\\ms_res.json'
        ws_old = f'data\\filter_tweets_num\\{disease}\\{id}\\ws_res.json'
        profile_old = f'data\\filter_tweets_num\\{disease}\\{id}\\profile.json'
        age_new = f'{profile_folder}\\age.json'
        gender_new = f'{profile_folder}\\gender.json'
        ms_new = f'{profile_folder}\\marital_status.json'
        ws_new = f'{profile_folder}\\work_status.json'
        shutil.move(age_old, age_new)
        shutil.move(gender_old, gender_new)
        shutil.move(ms_old, ms_new)
        shutil.move(ws_old, ws_new)
        shutil.move(profile_old, profile_folder)

def remove_memory(disease):
    print(disease)
    id_path = f'data\\uid\\{disease}.json'
    id_list = read_json(id_path)  
    for id in id_list:
        print('-----------------------------------------')
        print(id)
        old_path_le = dict()
        old_path_symp = dict()
        new_path_le = dict()
        new_path_symp = dict()
        for le in life_events_list:
            old_path_le[le] = f'data\\filter_tweets_num\\{disease}\\{id}\\memory\\detail\\{le}.json'
            new_folder = f'data\\filter_tweets_num\\{disease}\\{id}\\memory\\le'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)   
            new_path_le[le] = f'{new_folder}\\{le}.json'
            shutil.move(old_path_le[le], new_path_le[le])
        for symp in symptoms_list:
            old_path_symp[symp] = f'data\\filter_tweets_num\\{disease}\\{id}\\memory\\detail\\{symp}.json'
            new_folder = f'data\\filter_tweets_num\\{disease}\\{id}\\memory\\symp'
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)   
            new_path_symp[symp] = f'{new_folder}\\{symp}.json'
            shutil.move(old_path_symp[symp], new_path_symp[symp])
        os.rmdir(f'data\\filter_tweets_num\\{disease}\\{id}\\memory\\detail')    
    
    
def detect_gender(disease):
    print(disease)
    id_path = f'data\\uid\\{disease}.json'
    id_list = read_json(id_path)  
    for id in id_list:
        print('-----------------------------------------')
        print(id)
        ind = id_list.index(id)
        user_folder = f'data\\filter_tweets_num\\{disease}\\{id}'
        ms_path = f'{user_folder}\\marital_status_tweets.json'
        write_path = f'{user_folder}\\gender_res.json'
        des_path = f'data\\org\\{disease}\\{id}\\user.json'
        des_data = read_json(des_path)
        profile_des = des_data['description']
        mst = read_json(ms_path)
        texts = ''
        tweet_num = len(mst)
        for tweet in mst:
            text = tweet['text']
            texts = texts + text
        result = dict()
        model_gender = process_gender(disease, ind)
        result['model'] = model_gender
        keywords_gender = extract_gender(tweet_num, texts, profile_des)
        result['keywords'] = keywords_gender
        write_json(write_path, result)

def match(disease, mode, threshold):
    print(f'-----------------------------------------{disease}--------------------------------------------------')
    id_path = f'data/uid/{disease}.json'
    id_list = read_json(id_path)
    if mode == 'le':
        res_path = f'output\\le\\{disease}\\test.npy'
        column_names = ['Career','Death','Education','Financial','Health','Identity','Legal','Lifestyle_Change','New_Birth_in_Family',
                        'Relationships_Changes','Relocation','Societal']
    if mode == 'symp':
        res_path = f'output\\symp\\{disease}\\test.npy'
        column_names = ['Anxious_Mood','Autonomic_symptoms','Cardiovascular_symptoms','Catatonic_behavior',
                        'Decreased_energy_tiredness_fatigue','Depressed_Mood',
                        'Gastrointestinal_symptoms','Genitourinary_symptoms','Hyperactivity_agitation',
                        'Impulsivity','Inattention','Indecisiveness','Respiratory_symptoms','Suicidal_ideas',
                        'Worthlessness_and_guilty','avoidance_of_stimuli','compensatory_behaviors_to_prevent_weight_gain','compulsions',
                        'diminished_emotional_expression','do_things_easily_get_painful_consequences','drastical_shift_in_mood_and_energy',
                        'fear_about_social_situations','fear_of_gaining_weight','fears_of_being_negatively_evaluated','flight_of_ideas',
                        'intrusion_symptoms','loss_of_interest_or_motivation','more_talktive','obsession','panic_fear','pessimism','poor_memory',
                        'sleep_disturbance','somatic_muscle','somatic_symptoms_others','somatic_symptoms_sensory','weight_and_appetite_change','Anger_Irritability']
    result = np.load(res_path)
    row = 0
    for id in id_list:
        print(f'-------------{id}------------------')
        tweets_path = f'data\\filter_tweets_num\\{disease}\\{id}\\tweets.json'
        tweets = read_json(tweets_path)
        for date_tweet in tweets:
            for tweet in date_tweet:
                if mode == 'le':
                    tweet['life_events'] = []
                    modify_list = tweet['life_events']
                elif mode == 'symp':
                    tweet['symptoms'] = []
                    modify_list = tweet['symptoms']
                row_data = result[row]
                exceeding_columns = [
                    column_names[i] for i, value in enumerate(row_data) if value > threshold
                ]
                for col in exceeding_columns:
                    modify_list.append(col)
                row += 1
        write_json(tweets_path, tweets)

def filter_tweets(disease):
    print(disease)
    id_path = f'data/uid/{disease}.json'
    id_list = read_json(id_path)
    for id in id_list:
        print(f'----------------------{id}--------------------------')
        gros_dict = dict()
        total_count = 0
        days_count = 0
        le_total = 0
        symp_total = 0
        tweets_filtered = dict()
        tweets_path = f'data/filter_tweets_num/{disease}/{id}/tweets.json'
        write_path = f'data/filter_tweets_num/{disease}/{id}/tweets_filtered.json'
        info_path = f'data/filter_tweets_num/{disease}/{id}/tweets_info.json'
        detail_folder_le = f'data/filter_tweets_num/{disease}/{id}/memory/le'
        detail_folder_symp = f'data/filter_tweets_num/{disease}/{id}/memory/symp'
        if not os.path.exists(detail_folder_le):
            os.makedirs(detail_folder_le)  
        if not os.path.exists(detail_folder_symp):
            os.makedirs(detail_folder_symp)  
        tweets = read_json(tweets_path)
        path_le = dict()
        path_symp = dict()
        data_le = dict()
        data_symp = dict()
        le_detail = dict()
        symp_detail = dict()
        for ind, l_e in enumerate(life_events_list):
            path_le[l_e] = f'{detail_folder_le}/{l_e}.json'
            data_le[l_e] = []
        for ind, sy in enumerate(symptoms_list):
            path_symp[sy] = f'{detail_folder_symp}/{sy}.json'
            data_symp[sy] = []
        tweet_date = dict()
        for date_tweets in tweets:
            timestamp = date_tweets[0]['timestamp_tweet']
            date, _ = timestamp.split()
            tweets_list_new = []
            detail_dict = dict()
            tweets_num = 0
            le_count = 0
            symp_count = 0
            for tweet in date_tweets:
                life_events = tweet['life_events']
                symptoms = tweet['symptoms']
                if (life_events != []) or (symptoms != []):
                    keys_list = (detail_dict.keys())
                    tweets_num += 1
                    tweets_list_new.append(tweet)
                    for le in life_events:
                        if le and (le not in keys_list):
                            detail_dict[le] = 1
                            le_count += 1
                            data_le[le].append(tweet)
                        elif le and (le in keys_list):
                            detail_dict[le] += 1
                            le_count += 1
                            data_le[le].append(tweet)
                    for symp in symptoms:
                        if symp and (symp not in keys_list):
                            detail_dict[symp] = 1
                            symp_count += 1
                            data_symp[symp].append(tweet)
                        elif symp and (symp in keys_list):
                            detail_dict[symp] += 1 
                            symp_count += 1
                            data_symp[symp].append(tweet)
            if tweets_num != 0:
                days_count += 1
                tweets_filtered[date] = tweets_list_new
                tweet_date[date] = dict()
                tweet_date[date]['tweets_num'] = tweets_num
                tweet_date[date]['life_events'] = le_count
                tweet_date[date]['symptoms'] = symp_count
                tweet_date[date]['detail'] = detail_dict
            total_count += tweets_num
            le_total += le_count
            symp_total += symp_count
        sorted_detail_dict = {key: tweet_date[key] for key in sorted(tweet_date)}
        gros_dict['total_tweets'] = total_count
        gros_dict['tweet_days'] = days_count
        le_detail['total_counts'] = le_total
        symp_detail['total_counts'] = symp_total
        
        for key in data_le:
            write_json(path_le[key], data_le[key])
            le_detail[key] = len(data_le[key])
        for key in data_symp:
            write_json(path_symp[key], data_symp[key])
            symp_detail[key] = len(data_symp[key])
        
        gros_dict['life_events'] = le_detail
        gros_dict['symptoms'] = symp_detail
        
        gros_dict['details'] = sorted_detail_dict
        write_json(write_path, tweets_filtered)
        write_json(info_path, gros_dict)   
def add_sample_info(sample_path):
    sample_info = read_json(sample_path)
    id_dict = dict()
    for disease in diseases:
        id_dict[disease] = read_json(f'data\\uid\\{disease}.json')
    for class_sample in sample_info:
        class_res = sample_info[class_sample]
        for uid in class_res:
            # count = 0
            user_disease = None
            for d in id_dict:
                id_list = id_dict[d]
                if uid in id_list:
                    user_disease = d
            #         count += 1
            # if count != 1:
            #     print(f'{uid} replicate!')
            class_res[uid]['disease'] = user_disease
    write_json(sample_path, sample_info)
def process_sample(sample_path):
    sample_info = read_json(sample_path)
    info_dict = dict()
    for class_number in sample_info:
        class_dict = sample_info[class_number]
        for uid in class_dict:
            user_info = class_dict[uid]
            info_dict[uid] = user_info['disease']
    return info_dict       


def classify_work_status(disease, id_list):
    print(f'-----------------------------------------{disease}----------------------------------------')
    # id_path = f'data/uid/{disease}.json'
    # id_list = read_json(id_path)
    # id_list = ['630113']
    for id in id_list:
        print(f'-----------------{disease}--------------{id}-----------------')
        new_status = dict()
        ws_path = f'data\\filter_tweets_num\\{disease}\\{id}\\profile\\work_status.json'
        tweets_path = f'data\\filter_tweets_num\\{disease}\\{id}\\profile\\work_status_tweets.json'    
        ws = read_json(ws_path) 
        tweets = read_json(tweets_path)
        for tweet in tweets:
            tweet['text'] = clean_tweet(tweet['text'])
        career, work_status = classify_user(tweets, career_keywords, work_status_keywords)
        new_status['career'] = career
        new_status['work_status'] = work_status
        if ws != []:
            ws['similarity'] = new_status
        else:
            ws = dict()
            ws['similarity'] = new_status
        print(new_status)
        write_json(ws_path, ws)
        
def add_summary_to_profile(disease, id_list):
    print(f'-----------------------------------------{disease}----------------------------------------')
    # id_path = f'data/uid/{disease}.json'
    # id_list = read_json(id_path)
    # id_list = ['630113']
    for id in id_list:
        print(f'-----------------{disease}--------------{id}-----------------')
        read_path = f'data\\filter_tweets_num\\{disease}\\{id}\\memory\\conclusion.json'
        write_path = f'data\\filter_tweets_num\\{disease}\\{id}\\profile\\profile.json'
        cc = read_json(read_path)
        profile = read_json(write_path)
        le_dict = dict()
        symp_dict = dict()
        for item in cc:
            response = cc[item]
            if response:
                summary = response["conclusion"]
                if item in life_events_list:
                    le_dict[item] = summary
                elif item in symptoms_list:
                    symp_dict[item] = summary
        profile["life_events"] = le_dict
        profile["symptoms"] = symp_dict
        write_json(write_path, profile)


def count_tokens(text, model="gpt-3.5-turbo"):
    """
    计算给定文本的 token 数量。

    :param text: 要计算的文本
    :param model: 使用的模型名称（默认使用 gpt-3.5-turbo）
    :return: token 数量
    """
    # 获取对应模型的编码器
    encoding = tiktoken.encoding_for_model(model)
    
    # 将文本编码为 token
    tokens = encoding.encode(text)
    
    return len(tokens)

def simul_create(num):
    sim_folder = f'evaluation\\simul-{num}'
    sample_path = f'{sim_folder}\\sample.json'
    sample_res = read_json(sample_path)
    keys_list = list(sample_res.keys())
    for key in keys_list:
        class_folder = f'{sim_folder}\\user\\{key}'
        if not os.path.exists(class_folder):
            os.makedirs(class_folder) 
        sample_class = sample_res[key]
        if sample_class:
            id_list = list(sample_class.keys())
            for id in id_list:
                disease = sample_class[id]['disease']
                copy_folder(src=f'data\\filter_tweets_num\\{disease}\\{id}', dest=f'{class_folder}\\{id}')
                
def style_extract(id):
    data_path = f'experiment\\{id}\\tweets.json'
    data = read_json(data_path)
    write_path = f'experiment\\{id}\\style\\style_tweet.json'
    input_list = []
    with open("prompt\\style_extract.txt", encoding="utf-8", mode="r") as f:
        prompt = f.read()
    for date_tweets in data:
        for tweet in date_tweets:
            tweet_dict = dict()
            tweet_dict['text'] = tweet['text']
            tweet_dict['tweet_id'] = tweet['tweet_id']
            input_list.append(tweet_dict)
            if len(input_list) == 100:
                output_list = []
                variable_dict = {
                    "tweets" : input_list
                }
                prompt_text = load_variables_by_name(prompt, variable_dict)
                while True:
                    try:
                        prompt_text = load_variables_by_name(prompt, variable_dict)
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt_text}]
                        )
                        response = "{" + response.choices[0].message.content.split('{')[1].split('}')[0] + "}"
                        # response = client.chat.completions.create(
                        #     model="gpt-4o",
                        #     messages=[{"role": "user", "content": prompt_text}]
                        # )
                        print(prompt_text)
                        print(response)
                        # response = json.loads(response.choices[0].message.content)
                        answer = json.loads(response)
                        answer = answer["tweet_id"]
                        for id in answer:
                            for tweet_input in input_list:
                                if int(tweet_input['tweet_id']) == int(id):
                                    output_list.append(tweet_input)
                        break
                    except Exception as e:
                            print('Error message:', str(e))
                            continue
                with open(write_path, encoding="utf-8", mode="r") as f:
                    content = json.load(f)
                content.extend(output_list)
                with open(write_path, encoding="utf-8", mode="w") as f:
                    json.dump(content, f, ensure_ascii=False, indent=4)
                input_list = []
def style_condense(id):
    data_path = f'experiment\\{id}\\style\\style_tweet.json'
    date_tweets = read_json(data_path)
    write_path = f'experiment\\{id}\\style\\style_condense.json'
    input_list = []
    with open("prompt\\style_extract.txt", encoding="utf-8", mode="r") as f:
        prompt = f.read()
    for tweet in date_tweets:
        tweet_dict = dict()
        tweet_dict['text'] = tweet['text']
        tweet_dict['tweet_id'] = tweet['tweet_id']
        input_list.append(tweet_dict)
        if len(input_list) == 100:
            output_list = []
            variable_dict = {
                "tweets" : input_list
            }
            prompt_text = load_variables_by_name(prompt, variable_dict)
            while True:
                try:
                    prompt_text = load_variables_by_name(prompt, variable_dict)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt_text}]
                    )
                    response = "{" + response.choices[0].message.content.split('{')[1].split('}')[0] + "}"
                    # response = client.chat.completions.create(
                    #     model="gpt-4o",
                    #     messages=[{"role": "user", "content": prompt_text}]
                    # )
                    print(prompt_text)
                    print(response)
                    # response = json.loads(response.choices[0].message.content)
                    answer = json.loads(response)
                    answer = answer["tweet_id"]
                    for id in answer:
                        for tweet_input in input_list:
                            if int(tweet_input['tweet_id']) == int(id):
                                output_list.append(tweet_input)
                    break
                except Exception as e:
                        print('Error message:', str(e))
                        continue
            with open(write_path, encoding="utf-8", mode="r") as f:
                content = json.load(f)
            content.extend(output_list)
            with open(write_path, encoding="utf-8", mode="w") as f:
                json.dump(content, f, ensure_ascii=False, indent=4)
            input_list = []
def style_concon(id):
    data_path = f'experiment\\{id}\\style\\style_concon.json'
    date_tweets = read_json(data_path)
    write_path = f'experiment\\{id}\\style\\style_conconcon.json'
    input_list = []
    with open("prompt\\style_extract.txt", encoding="utf-8", mode="r") as f:
        prompt = f.read()
    for tweet in date_tweets:
        tweet_dict = dict()
        tweet_dict['text'] = tweet['text']
        tweet_dict['tweet_id'] = tweet['tweet_id']
        input_list.append(tweet_dict)
        output_list = []
        variable_dict = {
            "tweets" : input_list
        }
        prompt_text = load_variables_by_name(prompt, variable_dict)
        while True:
            try:
                prompt_text = load_variables_by_name(prompt, variable_dict)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_text}]
                )
                response = "{" + response.choices[0].message.content.split('{')[1].split('}')[0] + "}"
                # response = client.chat.completions.create(
                #     model="gpt-4o",
                #     messages=[{"role": "user", "content": prompt_text}]
                # )
                print(prompt_text)
                print(response)
                # response = json.loads(response.choices[0].message.content)
                answer = json.loads(response)
                answer = answer["tweet_id"]
                for id in answer:
                    for tweet_input in input_list:
                        if int(tweet_input['tweet_id']) == int(id):
                            output_list.append(tweet_input)
                break
            except Exception as e:
                    print('Error message:', str(e))
                    continue
        with open(write_path, encoding="utf-8", mode="r") as f:
            content = json.load(f)
        content.extend(output_list)
        with open(write_path, encoding="utf-8", mode="w") as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        input_list = []
def style_final(id):
    data_path = f'experiment\\{id}\\style\\style_concon.json'
    input_list = read_json(data_path)
    write_path = f'experiment\\{id}\\style\\style_conconcon.json'
    with open("prompt\\style_extract.txt", encoding="utf-8", mode="r") as f:
        prompt = f.read()
    output_list = []
    variable_dict = {
        "tweets" : input_list
    }
    prompt_text = load_variables_by_name(prompt, variable_dict)
    while True:
        try:
            prompt_text = load_variables_by_name(prompt, variable_dict)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_text}]
            )
            response = "{" + response.choices[0].message.content.split('{')[1].split('}')[0] + "}"
            print(prompt_text)
            print(response)
            answer = json.loads(response)
            answer = answer["tweet_id"]
            for id in answer:
                for tweet_input in input_list:
                    if int(tweet_input['tweet_id']) == int(id):
                        output_list.append(tweet_input)
            break
        except Exception as e:
                print('Error message:', str(e))
                continue
    with open(write_path, encoding="utf-8", mode="r") as f:
        content = json.load(f)
    content.extend(output_list)
    with open(write_path, encoding="utf-8", mode="w") as f:
        json.dump(content, f, ensure_ascii=False, indent=4)
    
    
# 评估采样质量
# 计算样本之间的平均距离
def avg_distance(vectors):
    distances = np.linalg.norm(np.array([vectors[i] - vectors[j] for i in range(len(vectors)) for j in range(len(vectors)) if i != j]), axis=1)
    return np.mean(distances)
# 计算特征方差
def feature_variances(vectors):
    return np.var(vectors, axis=0)

def sample():    
    for disease in diseases:
        data_dir = f'data\\user\\{disease}'
        # 用于存储所有用户的特征向量
        all_vectors = []
        all_ids = []
            
        # 文本向量化器
        text_vectorizer = TfidfVectorizer()
        # 数值特征标准化器
        num_scaler = StandardScaler()
        
        # 分批处理数据
        batch_size = 1000000
        current_batch = []
        current_batch_ids = []
        batch_count = 0
        
        for uid in os.listdir(data_dir):
            file_path = os.path.join(data_dir, uid, 'user.json')
            user_data = read_json(file_path)
            user_id = user_data['id']

            # 提取文本特征
            text_feature = user_data['description']

            # 提取数值特征
            num_feature = [
                user_data['favourites_count'],
                user_data['followers_count'],
                user_data['friends_count'],
                user_data['status_count'],
                int(user_data['verified_check']),
                user_data['tweet_info']['total_tweets'],
                user_data['tweet_info']['tweet_days'],
                user_data['tweet_info']['std_dev'],
                user_data['tweet_info']['life_events']['total_counts'],
                user_data['tweet_info']['symptoms']['total_counts']
            ]
            num_feature.extend(list(user_data['tweet_info']['life_events'].values())[1:])
            num_feature.extend(list(user_data['tweet_info']['symptoms'].values())[1:])

            current_batch.append((text_feature, num_feature))
            current_batch_ids.append(user_id)

            if len(current_batch) == batch_size:
                batch_texts, batch_nums = zip(*current_batch)

                # 文本向量化
                text_vectors = text_vectorizer.fit_transform(batch_texts).toarray()

                # 数值特征标准化
                num_vectors = num_scaler.fit_transform(batch_nums)

                # 合并特征
                batch_combined_vectors = np.hstack((text_vectors, num_vectors))

                for vec in batch_combined_vectors:
                    print(len(vec))
                    all_vectors.append(vec)
                
                # all_vectors.extend(batch_combined_vectors)
                all_ids.extend(current_batch_ids)

                current_batch = []
                current_batch_ids = []
                batch_count += 1

        # 处理剩余数据
        if current_batch:
            batch_texts, batch_nums = zip(*current_batch)

            # 文本向量化
            text_vectors = text_vectorizer.fit_transform(batch_texts).toarray()

            # 数值特征标准化
            num_vectors = num_scaler.fit_transform(batch_nums)
            
            # 合并特征
            batch_combined_vectors = np.hstack((text_vectors, num_vectors))

            for vec in batch_combined_vectors:
                print(len(vec))
                all_vectors.append(vec)
            # all_vectors.extend(batch_combined_vectors)
            all_ids.extend(current_batch_ids)

        all_vectors = np.array(all_vectors)
        all_ids = np.array(all_ids)

        # 使用UMAP降维
        umap_model = umap.UMAP(n_components=2)
        umap_vectors = umap_model.fit_transform(all_vectors)

        # 使用核密度估计
        kde = gaussian_kde(umap_vectors.T)
        densities = kde(umap_vectors.T)

        # 采样1000个用户
        sample_size = 1000
        # 基于密度进行概率采样
        probabilities = densities / np.sum(densities)
        sampled_indices = np.random.choice(len(all_vectors), size=sample_size, p=probabilities, replace=False)
        sampled_vectors = all_vectors[sampled_indices]
        sampled_ids = all_ids[sampled_indices]


        # 保存用户向量至文件
        np.save(f'{data_dir}\\sampled_vectors.npy', sampled_vectors)
        np.save(f'{data_dir}\\sampled_ids.npy', sampled_ids)
        # original_avg_dist = avg_distance(all_vectors)
        # sampled_avg_dist = avg_distance(sampled_vectors)
        # print(f"原始数据平均距离: {original_avg_dist}")
        # print(f"采样数据平均距离: {sampled_avg_dist}")
        original_variances = feature_variances(all_vectors)
        sampled_variances = feature_variances(sampled_vectors)

        print(f"原始数据特征方差: {original_variances}")
        print(f"采样数据特征方差: {sampled_variances}")

def create_sample():
    for disease in diseases:
        disease_folder = f'data\\user\\{disease}'
        id_path = f'{disease_folder}\\sampled_ids.npy'
        sample_id = read_npy(id_path)
        for id in sample_id:
            src_folder = f'{disease_folder}\\{id}'
            dst_foldler = f'sample\\{disease}\\{id}'
            copy(src_folder, dst_foldler)



            
        
        
        
if __name__ == "__main__":
    create_sample()
    
    # sample()
    # style_final(630113)
    # The above code is creating a dictionary in Python called `simul1_dict` with keys as numerical
    # IDs and values as strings representing different mental health conditions.
    # simul1_dict = {
    #     "29821526": "anxiety",
    #     "259857436": "ocd",
    #     "294269531": "depression",
    #     "418119964": "neg",
    #     "438472638": "neg",
    #     "454916594": "anxiety",
    #     "566132717": "anxiety",
    #     "801542352": "ptsd",
    #     "981193782": "depression"
    # }
    # simul1_dict = {
    #     "1055008202": "depression",
    #     "38173093": "anxiety",
    #     "147350203": "neg",
    #     "1313631394808160259": "adhd",
    #     "16313943": "anxiety",
    #     "25195823": "neg",
    #     "50883315": "adhd",
    #     "2349691311": "anxiety",
    #     "209964615": "anxiety"
    # }
    # id_list = list(simul1_dict.keys())
    # for id in id_list:
    #     input_list = [id]
    #     disease = simul1_dict[id]
    #     conclude_tweets(disease, input_list)
    #     classify_work_status(disease, input_list)
    #     add_summary_to_profile(disease, input_list)
    #     copy_folder(src=f"data\\filter_tweets_num\\{disease}\\{id}", dest="evaluation\\simul-1\\user")
    # sim_folder = 'evaluation\\simul-1'
    # sample_path = f'{sim_folder}\\sample.json'
    # sample_res = read_json(sample_path)
    # keys_list = list(sample_res.keys())
    # for key in keys_list:
    #     sample_class = sample_res[key]
    #     if sample_class:
    #         id_list = list(sample_class.keys())
    #         for id in id_list:
    #             disease = sample_class[id]['disease']
    #             id_l = [id]
    #             conclude_tweets(disease, id_l)
    #             # classify_work_status(disease, id_l)
    #             add_summary_to_profile(disease, id_l)
# add_summary_to_profile('depression')       
# classify_work_status('adhd')
# classify_work_status('anxiety')
# classify_work_status('bipolar')
# classify_work_status('depression')
# classify_work_status('neg')
# classify_work_status('ocd')
# classify_work_status('ptsd')
        
                
# add_sample_info('evaluation\question\simul-1\sample.json') 

# info_dict = process_sample('evaluation\simul-1\sample.json')
# for id in info_dict: 
#     print(f'----------------------{id}--------------------------')
#     user_folder = f"data\\filter_tweets_num\\{info_dict[id]}\\{id}"
#     copy(user_folder, f'copy\\data\\filter_tweets_num\\{info_dict[id]}\\{id}')

# work_status_tweet('adhd')
# work_status_tweet('anxiety')
# work_status_tweet('bipolar')
# work_status_tweet('depression')
# work_status_tweet('neg')
# work_status_tweet('ocd')
# work_status_tweet('ptsd')
    
        
# def copy_user(disease):
#     print(disease)
#     id_path = f'data/uid/{disease}.json'
#     id_list = read_json(id_path)
#     for id in id_list:
#         print(f'----------------------{id}--------------------------')
#         source_path1 = f'data\\org\\{disease}\\{id}\\user.json'
#         dst_folder = f'data\\filter_tweets_num\\{disease}\{id}'
#         shutil.copy(source_path1, dst_folder)

# def construct_memory(disease, version):
    # clean tweets label之间的相关性？
    # version one : unfiltered tweets
    # version two : filtered with life events and symptoms
    # version three : structured tweets filtered with life events and symptoms
    # version four : tweets filtered with topic
    # version five: system prompt



# copy_user('adhd')
# copy_user('anxiety')
# copy_user('bipolar')
# copy_user('depression')
# copy_user('neg')
# copy_user('ocd')
# copy_user('ptsd')

# filter_tweets('adhd')
# filter_tweets('anxiety')
# filter_tweets('bipolar')
# filter_tweets('depression')
# filter_tweets('neg')
# filter_tweets('ocd')
# filter_tweets('ptsd')

# match(disease='adhd', mode='le', threshold=0.5)  
# match(disease='adhd', mode='symp', threshold=0.5) 
# match(disease='anxiety', mode='le', threshold=0.5)  
# match(disease='anxiety', mode='symp', threshold=0.5) 
# match(disease='bipolar', mode='le', threshold=0.5)  
# match(disease='bipolar', mode='symp', threshold=0.5) 
# match(disease='depression', mode='le', threshold=0.5)  
# match(disease='depression', mode='symp', threshold=0.5) 
# match(disease='neg', mode='le', threshold=0.5)  
# match(disease='neg', mode='symp', threshold=0.5) 
# match(disease='ocd', mode='le', threshold=0.5)  
# match(disease='ocd', mode='symp', threshold=0.5) 
# match(disease='ptsd', mode='le', threshold=0.5)  
# match(disease='ptsd', mode='symp', threshold=0.5)


# clear_file('adhd')
# clear_file('anxiety')
# clear_file('bipolar')
# clear_file('depression')
# clear_file('neg')
# clear_file('ocd')
# clear_file('ptsd')

# for d in diseases:
#     remove_memory(d)        
        
# filter_tweets('adhd')
# filter_tweets('adhd')
# filter_tweets('adhd')
# filter_tweets('adhd')
# filter_tweets('adhd')
# filter_tweets('adhd')
# filter_tweets('adhd')
        
# for d in diseases:
#     marital_status_tweet(d)
# work_status_tweet('ptsd')
# construct_profile('adhd')
# construct_profile('anxiety')
# construct_profile('bipolar')
# construct_profile('depression')
# construct_profile('neg')
# construct_profile('ocd')
# construct_profile('ptsd')
# detect_gender('adhd')
# detect_gender('anxiety')
# detect_gender('bipolar')
# detect_gender('depression')
# detect_gender('neg')
# detect_gender('ocd')
# detect_gender('ptsd')