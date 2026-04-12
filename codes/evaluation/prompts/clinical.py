import json
import os

SYMPTOM_TIMELINE_DIR = "/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/data/stmhd_symptom_timeline"


CLINICAL_PROMPTS = {
  "任务-睡眠-睡眠浅": {
    "positive": "睡眠极浅，对声光敏感，夜间易惊醒且难再入睡。",
    "negative": "睡眠深沉，不易受环境干扰，能维持连续睡眠。"
  },
  "任务-睡眠-入睡困难": {
    "positive": "上床后辗转反侧，入睡潜伏期超过30分钟。",
    "negative": "上床后能迅速入睡，无入睡障碍。"
  },
  "任务-睡眠-多梦": {
    "positive": "梦境频繁或多噩梦，醒后疲惫感强。",
    "negative": "少梦或无梦，睡眠安稳。"
  },
  "任务-睡眠-早醒": {
    "positive": "比平日早醒1-2小时以上，醒后无法回笼，情绪恶劣。",
    "negative": "能睡至预定时间自然醒，无病理性早醒。"
  },
  "任务-睡眠-存在睡眠问题": {
    "positive": "整体睡眠质量差，无法通过睡眠恢复精力。",
    "negative": "睡眠质量良好，醒后精力充沛。"
  },
  "任务-睡眠-睡眠时间少": {
    "positive": "总睡眠时长显著缩短，远低于生理需求。",
    "negative": "睡眠时长充足，符合正常作息。"
  },
  "任务-食欲-暴饮暴食": {
    "positive": "发作性不可控地大量进食，常伴事后悔恨。",
    "negative": "饮食有节制，无暴食行为。"
  },
  "任务-食欲-显著体重变化": {
    "positive": "非刻意情况下，近期体重出现显著增减。",
    "negative": "近期体重保持相对稳定。"
  },
  "任务-食欲-食欲下降": {
    "positive": "无饥饿感，对食物丧失兴趣，进食量明显减少。",
    "negative": "胃口正常，按时按量进食。"
  },
  "任务-食欲-食欲存在问题": {
    "positive": "存在暴食或厌食等进食紊乱行为。",
    "negative": "饮食习惯规律正常。"
  },
  "任务-自杀-有无望感": {
    "positive": "对未来彻底绝望，坚信现状无法改善。",
    "negative": "对未来抱有希望，认为困难是暂时的。"
  },
  "任务-自杀-存在自杀倾向": {
    "positive": "反复出现结束生命的念头或具体计划。",
    "negative": "珍视生命，无轻生念头。"
  },
  "任务-自杀-自我价值感低": {
    "positive": "极度自卑，认为自己毫无价值，是他人的累赘。",
    "negative": "自我评价客观，认可自身价值。"
  },
  "任务-自杀-自罪": {
    "positive": "毫无根据地过度自责，产生病理性的负罪感。",
    "negative": "归因合理，不盲目揽责。"
  },
  "任务-自杀-存在自残倾向": {
    "positive": "有通过伤害身体来缓解精神痛苦的冲动或行为。",
    "negative": "无自伤意愿，懂得自我保护。"
  },
  "任务-自杀-存在自杀行为": {
    "positive": "近期实施过自杀尝试或处于准备阶段。",
    "negative": "从未实施过自杀行为。"
  },
  "任务-筛查-躁狂": {
    "positive": "曾有情绪持续高涨、精力过剩、思维奔逸的病史。",
    "negative": "情绪及精力水平始终处于正常区间。"
  },
  "任务-筛查-遗传史": {
    "positive": "直系或旁系亲属有精神疾病确诊史。",
    "negative": "家族无已知精神疾病史。"
  },
  "任务-躯体症状-躯体不适": {
    "positive": "伴有查无实据的疼痛、胸闷、头晕等躯体症状。",
    "negative": "身体无明显不明原因的不适感。"
  },
  "任务-躯体症状-运动性激越": {
    "positive": "烦躁不安，无法静坐，伴有搓手、踱步等小动作。",
    "negative": "肢体平静，能安静独处。"
  },
  "任务-躯体症状-运动性迟滞": {
    "positive": "思维迟缓，肢体沉重，行动及反应显著变慢。",
    "negative": "思维敏捷，行动利落。"
  },
  "任务-情绪-情绪低落": {
    "positive": "心境显著低落，感到悲伤、压抑或空虚。",
    "negative": "心境平稳或愉快。"
  },
  "任务-情绪-情绪低落超过两周": {
    "positive": "低落心境持续时间已达两周以上。",
    "negative": "情绪波动短暂，未达两周标准。"
  },
  "任务-情绪-早晚差异": {
    "positive": "情绪呈现“晨重暮轻”的节律变化。",
    "negative": "情绪全天平稳，无特定节律。"
  },
  "任务-兴趣-兴趣丧失": {
    "positive": "对既往爱好及日常活动完全丧失兴趣。",
    "negative": "保有好奇心，能从活动中获得乐趣。"
  },
  "任务-兴趣-范围-所有事情": {
    "positive": "兴趣丧失泛化至生活所有领域，无一例外。",
    "negative": "仍保留至少部分领域的兴趣。"
  },
  "任务-兴趣-情感淡漠": {
    "positive": "情感反应迟钝，对周围事物漠不关心。",
    "negative": "情感反应鲜活，能正常共情。"
  },
  "任务-兴趣-原因": {
    "positive": "兴趣丧失有明确的生活事件作为诱因。",
    "negative": "兴趣丧失无明显诱因，潜移默化发生。"
  },
  "任务-兴趣-兴趣丧失超过两周": {
    "positive": "兴趣缺乏的状态持续已逾两周。",
    "negative": "仅短暂感到无聊，未持续存在。"
  },
  "任务-兴趣-范围-过去爱好": {
    "positive": "特指对以往热衷的特定爱好失去热情。",
    "negative": "仍能从既往爱好中获得满足感。"
  },
  "闲聊-寻求帮助-询问医生的看法": {
    "positive": "主动征求诊断意见及治疗建议。",
    "negative": "被动接受问诊，不主动寻求建议。"
  },
  "闲聊-提供信息-被动提供相关信息": {
    "positive": "问一句答一句，甚至缄默，需反复引导。",
    "negative": "沟通顺畅，主动配合提供信息。"
  },
  "闲聊-提供信息-主动提供相关信息": {
    "positive": "积极倾诉症状细节及内心感受。",
    "negative": "仅限于回答既定问题，不主动发散。"
  },
  "闲聊-自我表露-抱怨自我": {
    "positive": "频繁自我贬低，抱怨自身能力或性格缺陷。",
    "negative": "自我陈述客观，无明显抱怨倾向。"
  },
  "任务-社会功能-日常生活存在困难": {
    "positive": "洗漱、进食等基础生活自理能力显著受损。",
    "negative": "生活自理完全正常。"
  },
  "任务-社会功能-学习工作存在困难": {
    "positive": "无法维持正常的工作效率或学习状态。",
    "negative": "胜任当前学业或职业要求。"
  },
  "任务-精神状态-疲倦": {
    "positive": "持续性极度疲乏，休息无法缓解。",
    "negative": "精力尚可，休息后体力恢复正常。"
  },
  "任务-社会功能-避免与人接触": {
    "positive": "主动回避社交，自我封闭，拒绝接触亲友。",
    "negative": "维持正常人际交往，乐于社交。"
  },
  "闲聊-自我表露-对事物的情绪": {
    "positive": "对外界事物表现出强烈的主观情绪色彩。",
    "negative": "叙事客观冷静，情绪卷入度低。"
  },
  "任务-精神状态-记忆力下降": {
    "positive": "显著健忘，难以保留短期记忆。",
    "negative": "记忆清晰，无认知功能下降。"
  },
  "任务-精神状态-缺乏自信": {
    "positive": "自我效能感低，对自己处理事情的能力缺乏信心。",
    "negative": "自信适度，相信自己能解决问题。"
  },
  "任务-社会功能-避免从亲友处得到支持": {
    "positive": "刻意隐瞒病情，拒绝亲友帮助。",
    "negative": "愿意向支持系统寻求并接受帮助。"
  },
  "任务-精神状态-选择困难": {
    "positive": "对日常琐事无法做出决断，优柔寡断。",
    "negative": "决策果断，无选择障碍。"
  },
  "任务-精神状态-注意力不集中": {
    "positive": "注意力难以维持，思维易涣散。",
    "negative": "专注力正常，能长时间集中精神。"
  }
}


SYMPTOM_PAIRS = {
    "任务-精神状态-疲倦": "Decreased_energy_tiredness_fatigue",
    "任务-情绪-情绪低落": "Depressed_Mood",
    "任务-精神状态-注意力不集中": "Inattention",
    "任务-精神状态-选择困难": "Indecisiveness",
    "任务-自杀-存在自杀倾向": "Suicidal_ideas",
    "任务-自杀-自我价值感低": "Worthlessness_and_guilty",
    "任务-兴趣-情感淡漠": "diminished_emotional_expression",
    "任务-筛查-躁狂": "drastical_shift_in_mood_and_energy",
    "任务-兴趣-兴趣丧失": "loss_of_interest_or_motivation",
    "任务-自杀-有无望感": "pessimism",
    "任务-精神状态-记忆力下降": "poor_memory",
    "任务-睡眠-存在睡眠问题": "sleep_disturbance",
    "任务-躯体症状-躯体不适": "somatic_symptoms_sensory",
    "任务-躯体症状-运动性激越": "Hyperactivity_agitation",
    "任务-躯体症状-运动性迟滞": "Catatonic_behavior",
    "任务-社会功能-避免与人接触": "fear_about_social_situations",
    "任务-食欲-食欲存在问题": "weight_and_appetite_change",
    "任务-食欲-显著体重变化": "fear_of_gaining_weight"
}



def clinical_prompt(profile_template):
    """
    根据 profile 生成临床 prompt
    对于 positive 症状：
    - 如果症状在 timeline 中存在，则加入时间线信息
    - 如果不存在，则只使用原始的 positive 描述
    """
    # 获取 candidate_id 并加载 symptom timeline
    profile = list(profile_template.values())[0]
    profile_id = list(profile_template.keys())[0]
    candidate_id = profile["candidate_id"][0]['basic_id']
    timeline_path = os.path.join(SYMPTOM_TIMELINE_DIR, f"{profile_id}_{candidate_id}.json")
    
    # 尝试加载 timeline，如果不存在则为空
    symptom_timeline = {}
    if os.path.exists(timeline_path):
        with open(timeline_path, "r", encoding="utf-8") as f:
            symptom_timeline = json.load(f)
    
    # 获取 timeline 中的症状索引
    timeline_symptoms = {}
    if symptom_timeline:
        timeline_symptoms = symptom_timeline.get("graph", {}).get("index", {}).get("by_symptom", {})
    
    # 获取 timeline 中的 nodes（用于获取详细信息）
    timeline_nodes = {}
    if symptom_timeline:
        for node in symptom_timeline.get("graph", {}).get("nodes", []):
            timeline_nodes[node["id"]] = node
    
    # 处理 positive symptoms
    positive_prompts = []
    used_timeline_symptoms = set()  # 记录已经使用过的 timeline 症状
    
    for symptom in profile["positive_symptoms"]:
        base_prompt = CLINICAL_PROMPTS[symptom]["positive"]
        
        # 检查是否有对应的英文症状名
        en_symptom = SYMPTOM_PAIRS.get(symptom)
        
        if en_symptom and en_symptom in timeline_symptoms:
            # 症状在 timeline 中存在，加入时间线信息
            used_timeline_symptoms.add(en_symptom)
            node_ids = timeline_symptoms[en_symptom]
            timeline_info = []
            for node_id in node_ids:
                if node_id in timeline_nodes:
                    node = timeline_nodes[node_id]
                    triple = node.get("triple", "")
                    time_norm = node.get("time_norm", {})
                    relative_cn = time_norm.get("relative_cn", "")
                    
                    # 构建时间线描述
                    if triple and relative_cn:
                        timeline_info.append(f"{relative_cn}：{triple}")
            
            if timeline_info:
                prompt_with_timeline = f"{base_prompt}（时间线：{'；'.join(timeline_info)}）"
                positive_prompts.append(prompt_with_timeline)
            else:
                positive_prompts.append(base_prompt)
        else:
            # 症状不在 timeline 中，直接使用原始描述
            positive_prompts.append(base_prompt)
    
    # 处理 negative symptoms
    negative_prompts = [CLINICAL_PROMPTS[symptom]["negative"] for symptom in profile["negative_symptoms"]]
  
    # 处理 timeline 中未被索引的症状（不在 SYMPTOM_PAIRS 映射中的）
    extra_timeline_prompts = []
    for timeline_symptom, node_ids in timeline_symptoms.items():
        if timeline_symptom not in used_timeline_symptoms:
            # 这个 timeline 症状没有被使用过，加入到额外的时间线中
            timeline_info = []
            for node_id in node_ids:
                if node_id in timeline_nodes:
                    node = timeline_nodes[node_id]
                    triple = node.get("triple", "")
                    time_norm = node.get("time_norm", {})
                    relative_cn = time_norm.get("relative_cn", "")
                    
                    if triple and relative_cn:
                        timeline_info.append(f"{relative_cn}：{triple}")
            
            if timeline_info:
                extra_timeline_prompts.append(f"[{timeline_symptom}]：{'；'.join(timeline_info)}")
    
    # 组合所有 prompts
    positive_intro = "以下是该患者的阳性症状，部分症状存在时间点叙述，如被问及，请提及时间点。\n"
    negative_intro = "下面是该患者正常的情况描述，如果你被问及与该症状相关的情况，请你予以否认！\n"
    
    prompts = positive_intro + "\n".join(positive_prompts) 
    if extra_timeline_prompts:
        prompts += "\n\n其他症状的时间线信息：\n" + "\n".join(extra_timeline_prompts)
    prompts += "\n\n" + negative_intro + "\n".join(negative_prompts)
    return prompts

def clinical_prompt_no_timeline(profile_template):
    """
    根据 profile 生成临床 prompt
    对于 positive 症状：
    - 如果症状在 timeline 中存在，则加入时间线信息
    - 如果不存在，则只使用原始的 positive 描述
    """
    # 获取 candidate_id 并加载 symptom timeline
    profile = list(profile_template.values())[0]
    profile_id = list(profile_template.keys())[0]
    candidate_id = profile["candidate_id"][0]['basic_id']
    timeline_path = os.path.join(SYMPTOM_TIMELINE_DIR, f"{profile_id}_{candidate_id}.json")
    
    # 尝试加载 timeline，如果不存在则为空
    symptom_timeline = {}
    if os.path.exists(timeline_path):
        with open(timeline_path, "r", encoding="utf-8") as f:
            symptom_timeline = json.load(f)
    
    # 获取 timeline 中的症状索引
    timeline_symptoms = {}
    if symptom_timeline:
        timeline_symptoms = symptom_timeline.get("graph", {}).get("index", {}).get("by_symptom", {})
    
    # 获取 timeline 中的 nodes（用于获取详细信息）
    timeline_nodes = {}
    if symptom_timeline:
        for node in symptom_timeline.get("graph", {}).get("nodes", []):
            timeline_nodes[node["id"]] = node
    
    # 处理 positive symptoms
    positive_prompts = []
    used_timeline_symptoms = set()  # 记录已经使用过的 timeline 症状
    
    for symptom in profile["positive_symptoms"]:
        base_prompt = CLINICAL_PROMPTS[symptom]["positive"]
        positive_prompts.append(base_prompt)
    
    # 处理 negative symptoms
    negative_prompts = [CLINICAL_PROMPTS[symptom]["negative"] for symptom in profile["negative_symptoms"]]
  
    # 组合所有 prompts
    positive_intro = "以下是该患者的阳性症状，部分症状存在时间点叙述，如被问及，请提及时间点。\n"
    negative_intro = "下面是该患者正常的情况描述，如果你被问及与该症状相关的情况，请你予以否认！\n"
    
    prompts = positive_intro + "\n".join(positive_prompts) 
    prompts += "\n\n" + negative_intro + "\n".join(negative_prompts)
    return prompts

if __name__ == "__main__":
    profile_template = {"0423": {
        "cr_id": "29",
        "d4_id": "2276",
        "age": 17,
        "gender": "M",
        "marital_status": "single",
        "work_status": "student",
        "big_five": {
            "Openness": 5,
            "Conscientiousness": 3,
            "Extraversion": 2,
            "Agreeableness": 4,
            "Neuroticism": 6
        },
        "candidate_id": [
            {
                "basic_id": "540149476",
                "similarity": 0.9603920767980495,
                "symp_similarity": 0.6153846153846154
            }
        ],
        "positive_symptoms": [
            "任务-睡眠-多梦",
            "任务-社会功能-日常生活存在困难",
            "任务-精神状态-注意力不集中",
            "任务-躯体症状-运动性激越",
            "任务-情绪-早晚差异",
            "任务-躯体症状-躯体不适",
            "任务-情绪-情绪低落",
            "任务-精神状态-选择困难",
            "任务-精神状态-疲倦",
            "任务-自杀-存在自杀倾向",
            "任务-自杀-自我价值感低",
            "任务-睡眠-入睡困难",
            "任务-情绪-情绪低落超过两周",
            "任务-社会功能-学习工作存在困难",
            "任务-兴趣-范围-过去爱好",
            "任务-社会功能-避免从亲友处得到支持",
            "任务-精神状态-缺乏自信",
            "任务-躯体症状-运动性迟滞",
            "任务-精神状态-记忆力下降",
            "任务-食欲-暴饮暴食"
        ],
        "negative_symptoms": [
            "任务-食欲-显著体重变化"
        ],
        "summation": "病人最近情绪低落，认知功能受损，注意力，记忆力下降，兴趣减低，影响到了个人生活，精力不足，睡眠障碍，心烦，严重时坐立不安的，伴躯体不适感，头晕，诊断抑郁发作。",
        "depression_risk": 1,
        "suiside_risk": 1
    }}
    
    print(clinical_prompt(profile_template))