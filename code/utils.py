import json
import os
import random

import gradio as gr
import torch

from agent import TimelineAgent
from repo_paths import DATA_DIR

def load_symptom_timeline(profile, candidate_index, max_events_num = 50):
    """
    load symptom timeline from symptom_timeline dataset
    """
    if not profile['candidate_id']:
        return None
    timeline_agent = TimelineAgent(profile, candidate_index, "symptom")
    return timeline_agent.get_cut_timeline(max_events_num = max_events_num) # return the cut timeline

def load_life_event_timeline(profile, candidate_index, max_events_num = 50):
    """
    load life event timeline from life_event_timeline dataset
    """
    if not profile['candidate_id']:
        return None 
    timeline_agent = TimelineAgent(profile, candidate_index, "life_event")
    return timeline_agent.get_cut_timeline(max_events_num = max_events_num) # return the cut timeline


from repo_paths import DATA_DIR


def load_clinical_dialogues(profile, dir=None, max_dialogues_num=10):
    if dir is None:
        dir = str(DATA_DIR / "d4_dialogues")
    """
    load clinical dialogues from d4 dataset
    save both doctor and patient messages
    output the reference messages with max_dialogues_num dialogues
    """
    with open(os.path.join(dir, f"{profile['d4_id']}.json"), "r") as f:
        dialogues = json.load(f)
    max_dialogues_num = min(max_dialogues_num, len(dialogues)//2)
    random_indices = random.choices(range(len(dialogues)//2), k=max_dialogues_num)
    user_messages = [f"病人：{dialogue['content'].strip()}\n" for dialogue in dialogues if dialogue["role"] == "patient"]
    reference_messages = ""
    for i in random_indices:
        reference_messages += f"{dialogues[2*i]['role']}：{dialogues[2*i]['content'].strip()}\n"
        reference_messages += f"{dialogues[2*i+1]['role']}：{dialogues[2*i+1]['content'].strip()}\n"
    return reference_messages


def load_consultation_dialogues(profile, dir=None, max_dialogues_num=10):
    """
    load clinical dialogues from d4 dataset
    save both doctor and patient messages
    output the reference messages with max_dialogues_num dialogues
    """
    if dir is None:
        dir = str(DATA_DIR / "cr_dialogues")
    with open(os.path.join(dir, f"{profile['cr_id']}.json"), "r") as f:
        dialogues = json.load(f)
    max_dialogues_num = min(max_dialogues_num, len(dialogues)//2)
    random_indices = random.choices(range(len(dialogues)//2), k=max_dialogues_num)
    user_messages = [f"{dialogue['role']}：{dialogue['content'].strip()}\n" for dialogue in dialogues if dialogue["role"] == "patient"]
    reference_messages = ""
    for i in random_indices:
        reference_messages += f"{dialogues[2*i]['role']}：{dialogues[2*i]['content'].strip()}\n"
        reference_messages += f"{dialogues[2*i+1]['role']}：{dialogues[2*i+1]['content'].strip()}\n"
    return reference_messages

def generate_patient_prompts(profile,
                             language="zh"):
    """
    Generate system prompts for all baselines (G1~G6),
    with multi-language support ("zh" or "en").
    """
    
    # 语言输出模板
    if language == "zh":
        lang_tag = "请你始终用中文回答问题。"
        L = {
            "use_profile": "请严格根据以下人口学与人格信息来表现：",
            "no_other_info": "不要使用任何症状、时间线或临床信息。",
            "use_clinical": "请根据以下人口学、人格信息与临床特征来模拟：",
            "use_symptom_tl": "请根据以下症状时间线表达症状的时间变化：",
            "use_event_tl": "请根据以下生活事件时间线表达情绪与功能变化：",
            "use_partial": "请结合人口学/人格、临床信息与症状时间线来模拟：",
            "use_full": "请基于以下四类信息来模拟真实来访者：",
            "no_timeline": "不要使用任何症状时间线或生活事件。",
            "no_events": "不要使用任何生活事件。",
            "no_clinical": "不要使用临床总结。",
            "bio": "人口学与人格信息：",
            "clinical": "临床信息：",
            "symptom_tl_label": "症状时间线：",
            "event_tl_label": "生活事件时间线：",
            "full_must": "你的回答必须体现人格一致性、临床合理性、时间线变化与事件触发效应。"
        }

        L["simulate"] = """
        你正在模拟一位抑郁症相关的患者。
        - 你是来访者，不是助理；不要为了“配合完成问诊”而组织长解释。
        - 每轮最多2句；每句尽量不超过20个汉字；口语化。
        - 只回答被问到的点；不要主动一次讲完全部背景。
        - 不需要每次都表示理解/赞同/感谢；不确定就说“不知道/说不清”。
        """

        L["use_consultation_info"] = """
        咨询参考：请根据以下咨询对话信息来模拟抑郁症患者进行心理咨询时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“说话方式”（语气、句子长短、犹豫、回避、表达情绪的方式）。
        - 你的回答以“来访者视角”说话，优先短句、口语化；允许停顿、含糊、说不清。
        - 不要每句都配合或总结咨询师的话；被追问时才多说一点点。

        【示例对话】
        """

        L["use_assessment_info"] = """
        问诊参考：请根据以下问诊信息来模拟抑郁症患者与精神科医生对话时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“问诊场景的回答方式”：简短、对症状频率/程度的模糊描述、被追问再补充。
        - 只回答医生问到的点；不要主动长篇解释原因或讲完整故事。
        - 不确定就说“不确定/说不清/差不多”；必要时允许回避隐私问题。

        【示例对话】
        """
    else:  # English version
        lang_tag = "All your responses must be in English."
        L = {
            "simulate": "You are simulating a mental-health patient.",
            "use_profile": "Use ONLY the following demographic and personality profile:",
            "no_other_info": "Do NOT use any symptoms, timelines, or clinical information.",
            "use_clinical": "Use the demographic/personality profile AND the clinical information:",
            "use_symptom_tl": "Use the following symptom timeline to express temporal symptom changes:",
            "use_event_tl": "Use the following life-event timeline to express emotional/functional changes:",
            "use_partial": "Use demographic/personality profile, clinical information, AND symptom timeline:",
            "use_full": "Simulate a patient grounded in the following four information sources:",
            "no_timeline": "Do NOT use any symptom timeline or life events.",
            "no_events": "Do NOT use any life events.",
            "no_clinical": "Do NOT use clinical summary unless explicitly referenced.",
            "bio": "Demographic & Personality Profile:",
            "clinical": "Clinical Information:",
            "symptom_tl_label": "Symptom Timeline:",
            "event_tl_label": "Life Event Timeline:",
            "full_must": "Your responses must reflect personality consistency, clinical plausibility, temporal changes, and event-triggered variations."
        }

    # 格式化 Big Five
    bf = profile["big_five"]
    bf_text = ", ".join([f"{k} {v}" for k, v in bf.items()])

    basic_profile_text = (
        f"Age: {profile['age']}\n"
        f"Gender: {profile['gender']}\n"
        f"Marital status: {profile['marital_status']}\n"
        f"Work status: {profile['work_status']}\n"
        f"Big Five: {bf_text}\n"
    )

    # 临床症状文本
    clinical_text = (
        (L["clinical"] + "\n")
        + "Positive symptoms:\n  - "
        + "\n  - ".join(profile["positive_symptoms"])
        + "\n\nNegative symptoms:\n  - "
        + "\n  - ".join(profile["negative_symptoms"])
        + f"\n\nSummary:\n{profile['summation']}"
    )

    # 时间线文本
    symptom_timeline = load_symptom_timeline(profile, 0, max_events_num = 50)
    event_timeline = load_life_event_timeline(profile, 0, max_events_num = 50)
    symptom_tl_text = (
        f"{L['symptom_tl_label']}\n" +
        (symptom_timeline if symptom_timeline else "(No symptom timeline provided.)")
    )

    event_tl_text = (
        f"{L['event_tl_label']}\n" +
        (event_timeline if event_timeline else "(No life-event timeline provided.)")
    )

    prompts = {}

    # G1：人口信息学+大五人格+基础clinical
    prompts["G1"] = f"""/no_thinking{L['simulate']}
                    {L['use_profile']}
                    {basic_profile_text}

                    2) {L['clinical']}
                    {clinical_text}
                    """

    # G2：人口信息学+大五人格+基础clinical+问诊对话信息
    clinical_messages = load_clinical_dialogues(profile)
    prompts["G2"] = f"""/no_thinking{L['simulate']}
                    {L['use_profile']}
                    {basic_profile_text}

                    2) {L['clinical']}
                    {clinical_text}

                    3) {L['use_assessment_info']}
                    {clinical_messages}
                        """

    # G3：人口信息学+大五人格+基础clinical+咨询对话信息
    consultation_messages = load_consultation_dialogues(profile)
    prompts["G3"] = f"""/no_thinking{L['simulate']}
                    {L['use_profile']}
                    {basic_profile_text}

                    2) {L['clinical']}
                    {clinical_text}

                    3) {L['use_consultation_info']}
                    {consultation_messages}
                        """

    # G4
    
    prompts["G4"] = f"""/no_thinking{L['simulate']}
{L['use_event_tl']}
{basic_profile_text}

{event_tl_text}

{L['no_clinical']}
{lang_tag}
"""

    # G5
    prompts["G5"] = f"""/no_thinking{L['simulate']}
{L['use_partial']}
{basic_profile_text}

{clinical_text}

{symptom_tl_text}

{L['no_events']}
{lang_tag}
"""

    # G6
    prompts["G6"] = f"""/no_thinking{L['simulate']}
{L['use_full']}

1) {L['bio']}
{basic_profile_text}

2) {L['clinical']}
{clinical_text}

3) {L['symptom_tl_label']}
{symptom_tl_text}

4) {L['event_tl_label']}
{event_tl_text}

{L['full_must']}
{lang_tag}
"""

    return prompts


def build_messages(system_prompt, profile_text, history, query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{profile_text}"}
    ]

    # Assistant 接收 profile
    messages.append({"role": "assistant", "content": "好的，我已经记住自己的病人资料了，请医生开始问诊。"})

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": "/no_thinking " + query})

    return messages