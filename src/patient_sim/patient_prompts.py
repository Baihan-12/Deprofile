import json
import os
import random

from patient_sim.timeline_agent import TimelineAgent
from patient_sim.prompts.personality_traits import big_five_prompt
from patient_sim.prompts.risk import risk_prompt
from patient_sim.prompts.clinical import clinical_prompt, clinical_prompt_no_timeline

random.seed(42)


def _default_data_root() -> str:
    """Repository `data/` directory, overridable via DEPROFILE_DATA_ROOT."""
    return os.environ.get(
        "DEPROFILE_DATA_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
    )




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


def load_clinical_dialogues(profile_id, profile, dir=None, max_dialogues_num=10):
    """
    load clinical dialogues from d4 dataset
    save both doctor and patient messages
    output the reference messages with max_dialogues_num dialogues
    """
    if dir is None:
        dir = os.path.join(_default_data_root(), "dialogues", "assessment")
    # with open(os.path.join(dir, f"{profile['d4_id']}.json"), "r") as f:
    #     dialogues = json.load(f)
    with open(os.path.join(dir, f"{profile_id}.json"), "r") as f:
        dialogues = json.load(f)
    max_dialogues_num = min(max_dialogues_num, len(dialogues)//2)
    random_indices = random.choices(range(len(dialogues)//2), k=max_dialogues_num)
    other_indices = [i for i in range(len(dialogues)//2) if i not in random_indices]
    other_dialogues = []
    for i in other_indices:
        other_dialogues.append(dialogues[2*i])
        other_dialogues.append(dialogues[2*i+1])
    # with open(os.path.join(RANDOM_DIALOGUES_DIR+"/assessment", f"{profile_id}.json"), "w") as f:
    #     json.dump(other_dialogues, f, ensure_ascii=False, indent=4)
    reference_messages = ""
    for i in random_indices:
        reference_messages += f"{dialogues[2*i]['role']}：{dialogues[2*i]['content'].strip()}\n"
        reference_messages += f"{dialogues[2*i+1]['role']}：{dialogues[2*i+1]['content'].strip()}\n"
    return reference_messages

def load_consultation_dialogues(profile_id, profile, dir=None, max_dialogues_num=10):
    """
    load clinical dialogues from d4 dataset
    save both doctor and patient messages
    output the reference messages with max_dialogues_num dialogues
    """
    if dir is None:
        dir = os.path.join(_default_data_root(), "dialogues", "counseling")
    #  with open(os.path.join(dir, f"{profile['cr_id']}.json"), "r") as f:
    with open(os.path.join(dir, f"{profile_id}.json"), "r") as f:
        dialogues = json.load(f)
    max_dialogues_num = min(max_dialogues_num, len(dialogues)//2)
    random_indices = random.choices(range(len(dialogues)//2), k=max_dialogues_num)
    other_indices = [i for i in range(len(dialogues)//2) if i not in random_indices]
    other_dialogues = []
    for i in other_indices:
        other_dialogues.append(dialogues[2*i])
        other_dialogues.append(dialogues[2*i+1])
    # with open(os.path.join(RANDOM_DIALOGUES_DIR+"/counseling", f"{profile_id}.json"), "w") as f:
    #     json.dump(other_dialogues, f, ensure_ascii=False, indent=4)
    reference_messages = ""
    for i in random_indices:
        reference_messages += f"{dialogues[2*i]['role']}：{dialogues[2*i]['content'].strip()}\n"
        reference_messages += f"{dialogues[2*i+1]['role']}：{dialogues[2*i+1]['content'].strip()}\n"
    return reference_messages

def generate_patient_prompts_old(profile,
                             language="zh",
                             life_event_rendering_card = None,
                             symptom_rendering_card = None):
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
        - 口语化，整段不能超过100个汉字。
        - 只回答被问到的点；不要主动一次讲完全部背景。
        - 不需要每次都表示理解/赞同/感谢
        """

        L["use_consultation_info"] = """
        咨询参考：请根据以下咨询对话信息来模拟抑郁症患者进行心理咨询时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“说话方式”（语气、句子长短、犹豫、回避、表达情绪的方式）。


        【示例对话】
        """
        # - 你的回答以“来访者视角”说话，优先短句、口语化；允许停顿、含糊、说不清。
        # - 不要每句都配合或总结咨询师的话；被追问时才多说一点点。

        L["use_assessment_info"] = """
        问诊参考：请根据以下问诊信息来模拟抑郁症患者与精神科医生对话时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“问诊场景的回答方式”


        【示例对话】
        """
        # - 只回答医生问到的点；不要主动长篇解释原因或讲完整故事。
        # - 不确定就说“不确定/说不清/差不多”；必要时允许回避隐私问题。

        L["clinical"] = """
        下面是临床信息，你会获得患者的阳性症状和阴性症状，以及临床总结。
        你需要在被问及阳性症状时表示肯定，在被问及阴性症状时否认。
        """

        L["use_symptom_tl"] = """
        请根据以下【症状时间线卡片(cards)】表达症状的时间变化。
        要求：
        1) 只使用 cards 中出现的症状/变化信息，不要新增不存在的症状细节。
        2) 你每次提到一个症状变化（出现/加重/缓解/反复），必须明确说出对应卡片的【代表时间点】。
        3) 回答尽量口语自然、不要像在读列表；每次最多引用 1–2 张卡片。
        4) 如果医生追问“什么时候/持续多久/之前后来”，必须从 cards 中再选一张更早或更晚的卡片补充时间点。

        【症状时间线卡片】
        """

        L["use_event_tl"] = """
        请根据以下【生活事件时间线卡片(cards)】表达情绪与功能变化，并体现事件触发/影响。
        要求：
        1) 只使用 cards 中出现的事件与影响，不要编造具体原因/地点/人物。
        2) 你每次提到一个事件或其影响，必须明确说出对应卡片的【代表时间点】
        （格式：'X（Y天前）'）。
        3) 若表达“从那以后/因为/导致”，必须让因果链落在 cards 里（即事件与变化都来自 cards），否则不要写因果。
        4) 每次最多引用 1–2 张卡片，避免堆砌。

       【生活事件时间线卡片】
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

    positive_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["positive_symptoms"]])
    negative_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["negative_symptoms"]])
    clinical_text = (
        (L["clinical"] + "\n")
        + "这些是患者出现的阳性症状，请在被问到时肯定这些症状:\n  - "
        + positive_symptoms_text
        + "\n\n这些是患者没有出现的症状，请在被问到时否认这些症状:\n  - "
        + negative_symptoms_text
        + f"\n\n临床总结:\n{profile['summation']}"
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
    clinical_messages = load_clinical_dialogues(_dialogue_profile_id(profile), profile)
    prompts["G2"] = f"""/no_thinking{L['simulate']}
                    {L['use_profile']}
                    {basic_profile_text}

                    2) {L['clinical']}
                    {clinical_text}

                    3) {L['use_assessment_info']}
                    {clinical_messages}
                        """

    # G3：人口信息学+大五人格+基础clinical+咨询对话信息
    consultation_messages = load_consultation_dialogues(_dialogue_profile_id(profile), profile)
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
                    {L['use_profile']}
                    {basic_profile_text}

                    2) {L['clinical']}
                    {clinical_text}

                    3) {L['use_assessment_info']}
                    {clinical_messages}

                    4) {L['use_consultation_info']}
                    {consultation_messages}
                        """

    # G5
    if life_event_rendering_card:
        prompts["G5"] = prompts["G4"] + f"5) {L['use_event_tl']}{life_event_rendering_card}"
    else:
        prompts["G5"] = prompts["G4"]
        prompts["G5"] += f"""{L['use_event_tl']}{event_tl_text}"""

    # G6

    if symptom_rendering_card:
        prompts["G6"] = prompts["G4"] + f"5) {L['use_symptom_tl']}{symptom_rendering_card}"
    else:
        prompts["G6"] = prompts["G4"]
        prompts["G6"] += f"""{L['use_symptom_tl']}{symptom_tl_text}"""

    # G7
    if life_event_rendering_card and symptom_rendering_card:
        prompts["G7"] = prompts["G6"] + f"6) {L['use_event_tl']}{life_event_rendering_card}"
    else:
        prompts["G7"] = prompts["G6"]
        prompts["G7"] += f"""{L['use_event_tl']}{event_tl_text}"""


    return prompts

def generate_patient_prompts_v2(profile_id,
                             profile ,
                             language="zh",
                             life_event_rendering_card = None,
                             symptom_rendering_card = None):
    """
    Generate system prompts for all baselines (G1~G6),
    with multi-language support ("zh" or "en").
    """
    
    # 语言输出模板
    if language == "zh":
        lang_tag = "请你始终用中文回答问题。"
        L = {
            "use_profile": "请严格根据以下人口学信息来模拟患者：",
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
        - 每句尽量不超过20个汉字；口语化，整段不能超过100个汉字。
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

        L["clinical"] = """
        下面是临床信息，你会获得患者的阳性症状和阴性症状，以及临床总结。
        你需要在被问及阳性症状时表示肯定，在被问及阴性症状时否认。
        """

        L["use_symptom_tl"] = """
        请根据以下【症状时间线卡片(cards)】表达症状的时间变化。
        要求：
        1) 只使用 cards 中出现的症状/变化信息，不要新增不存在的症状细节。
        2) 你每次提到一个症状变化（出现/加重/缓解/反复），必须明确说出对应卡片的【代表时间点】
        （格式：'X（Y天前）'）。
        3) 回答尽量口语自然、不要像在读列表；每次最多引用 1–2 张卡片。
        4) 如果医生追问“什么时候/持续多久/之前后来”，必须从 cards 中再选一张更早或更晚的卡片补充时间点。

        【症状时间线卡片】
        """

        L["use_event_tl"] = """
        请根据以下【生活事件时间线卡片(cards)】表达情绪与功能变化，并体现事件触发/影响。
        要求：
        1) 只使用 cards 中出现的事件与影响，不要编造具体原因/地点/人物。
        2) 你每次提到一个事件或其影响，必须明确说出对应卡片的【代表时间点】
        （格式：'X（Y天前）'）。
        3) 若表达“从那以后/因为/导致”，必须让因果链落在 cards 里（即事件与变化都来自 cards），否则不要写因果。
        4) 每次最多引用 1–2 张卡片，避免堆砌。

       【生活事件时间线卡片】
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



    # B prompt
    basic_profile_text = (
        f"{L['use_profile']}\n"
        f"Age: {profile['age']}\n"
        f"Gender: {profile['gender']}\n"
        f"Marital status: {profile['marital_status']}\n"
        f"Work status: {profile['work_status']}\n"
        )
    
    # P personality traits profile
    personality_traits_text = big_five_prompt(profile)
    

    # S1： symptom attributes
    positive_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["positive_symptoms"]])
    negative_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["negative_symptoms"]])
    # clinical_text = (
    #     (L["clinical"] + "\n")
    #     + "这些是患者出现的阳性症状，请在被问到时肯定这些症状:\n  - "
    #     + positive_symptoms_text
    #     + "\n\n这些是患者没有出现的症状，请在被问到时否认这些症状:\n  - "
    #     + negative_symptoms_text
    #     + f"\n\n临床总结:\n{profile['summation']}"
    # )

    clinical_text = clinical_prompt(profile) + f"\n\n临床总结:\n{profile['summation']}"

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

    prompts["G0"] = f"""/no_thinking {lang_tag}
                    {L['simulate']}
                    {L['use_profile']}
                    {basic_profile_text}
                    {risk_prompt(profile)}
                    """
    # G1：人口信息学+大五人格+基础clinical
    prompts["G1"] = prompts["G0"] + personality_traits_text


    prompts["G2-1"] = prompts["G1"] + f"2) {L['clinical']} {clinical_text}"

    # G2：人口信息学+大五人格+基础clinical+问诊对话信息
    clinical_messages = load_clinical_dialogues(profile_id, profile)
    prompts["G3-1"] = prompts["G2-1"] + f"3) {L['use_assessment_info']} {clinical_messages}"

    # G4：人口信息学+大五人格+基础clinical+咨询对话信息
    consultation_messages = load_consultation_dialogues(profile_id, profile)
    prompts["G4-1"] = prompts["G2-1"] + f"3) {L['use_consultation_info']} {consultation_messages}"

    # G5
    
    prompts["G5-1"] = prompts["G4-1"] + f"4) {L['use_assessment_info']} {clinical_messages}"
    # G6
    if life_event_rendering_card:
        prompts["G6-1"] = prompts["G5-1"] + f"5) {L['use_event_tl']}{life_event_rendering_card}"
    else:
        prompts["G6-1"] = prompts["G5-1"]
        prompts["G6-1"] += f"""{L['use_event_tl']}{event_tl_text}"""

    # G2-2
    if symptom_rendering_card:
        prompt_symp = symptom_rendering_card
    else:
        prompt_symp = f"{L['use_symptom_tl']}{symptom_tl_text}"


    prompts["G2-2"] = prompts["G1"] + f"2) {prompt_symp}"

    prompts["G3-2"] = prompts["G2-2"] + f"3) {L['use_assessment_info']} {clinical_messages}"
    
    prompts["G4-2"] = prompts["G3-2"] + f"3) {L['use_consultation_info']} {consultation_messages}"
    prompts["G5-2"] = prompts["G4-2"] + f"4) {L['use_assessment_info']} {clinical_messages}"
    if life_event_rendering_card:
        prompts["G6-2"] = prompts["G5-2"] + f"5) {L['use_event_tl']}{life_event_rendering_card}"
    else:
        prompts["G6-2"] = prompts["G5-2"]
        prompts["G6-2"] += f"5) {L['use_event_tl']}{event_tl_text}"


    return prompts

def generate_patient_prompts(profile_template,
                             language="zh",
                             life_event_rendering_card = None,
                             symptom_rendering_card = None):
    """
    Generate system prompts for all baselines (G1~G6),
    with multi-language support ("zh" or "en").
    """
    profile = list(profile_template.values())[0]
    profile_id = list(profile_template.keys())[0]
    # 语言输出模板
    if language == "zh":
        lang_tag = "请你始终用中文回答问题。"
        L = {
            "use_profile": "请严格根据以下人口学信息来模拟患者：",
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
        - 口语化，整段不能超过100个汉字。

        """
        # - 只回答被问到的点；不要主动一次讲完全部背景。
        # - 不需要每次都表示理解/赞同/感谢；不确定就说“不知道/说不清”。
        L["use_consultation_info"] = """
        咨询参考：请根据以下咨询对话信息来模拟抑郁症患者进行心理咨询时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“说话方式”（语气、句子长短、犹豫、回避、表达情绪的方式）。
        - 你的回答以“来访者视角”说话，优先短句、口语化；允许停顿、含糊、说不清。


        【示例对话】
        """
        # - 不要每句都配合或总结咨询师的话；被追问时才多说一点点。

        L["use_assessment_info"] = """
        问诊参考：请根据以下问诊信息来模拟抑郁症患者与精神科医生对话时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“问诊场景的回答方式”：简短、对症状频率/程度的模糊描述、被追问再补充。
        - 只回答医生问到的点

        【示例对话】
        """

        #         - 只回答被问到的点；不要主动一次讲完全部背景。
        # - 不需要每次都表示理解/赞同/感谢；不确定就说“不知道/说不清”。

        L["clinical"] = """
        下面是临床信息，你会获得患者的阳性症状和阴性症状，以及临床总结。
        你需要在被问及阳性症状时表示肯定，在被问及阴性症状时否认。
        """



        L["use_event_cards"] = """
        请根据以下生活事件发帖，并体现事件触发/影响。
        要求：
        1) 只使用生活事件时间线中出现的事件与影响，不要编造具体原因/地点/人物。
        2) 你每次提到一个事件或其影响，必须明确说出对应时间点
        3) 若表达“从那以后/因为/导致”，必须让因果链落在生活事件时间线中（即事件与变化都来自生活事件时间线），否则不要写因果。
        4) 每次最多引用 1–2 个事件，避免堆砌， 尽量口语化，不要像在读列表。

       【生活事件时间线】
        """

        L["use_event_tl"] = """
        请根据以下【生活事件时间线卡片(cards)】表达情绪与功能变化，并体现事件触发/影响。
        要求：
        1) 只使用 cards 中出现的事件与影响，不要编造具体原因/地点/人物。
        2) 你每次提到一个事件或其影响，必须明确说出对应卡片的时间点
        3) 若表达“从那以后/因为/导致”，必须让因果链落在 cards 里（即事件与变化都来自 cards），否则不要写因果。
        4) 每次最多引用 1–2 张卡片，避免堆砌， 尽量口语化，不要像在读列表。

       【生活事件时间线卡片】
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



    # B prompt
    basic_profile_text = (
        f"{L['use_profile']}\n"
        f"Age: {profile['age']}\n"
        f"Gender: {profile['gender']}\n"
        f"Marital status: {profile['marital_status']}\n"
        f"Work status: {profile['work_status']}\n"
        )
    
    # P personality traits profile
    personality_traits_text = big_five_prompt(profile)
    

    # S1： symptom attributes
    positive_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["positive_symptoms"]])
    negative_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["negative_symptoms"]])
    # clinical_text = (
    #     (L["clinical"] + "\n")
    #     + "这些是患者出现的阳性症状，请在被问到时肯定这些症状:\n  - "
    #     + positive_symptoms_text
    #     + "\n\n这些是患者没有出现的症状，请在被问到时否认这些症状:\n  - "
    #     + negative_symptoms_text
    #     + f"\n\n临床总结:\n{profile['summation']}"
    # )

    clinical_text = clinical_prompt(profile_template) + f"\n\n临床总结:\n{profile['summation']}"

    # 时间线文本
    # event_timeline = load_life_event_timeline(profile, 0, max_events_num = 50)
    # event_tl_text = (
    #     f"{L['event_tl_label']}\n" +
    #     (event_timeline if event_timeline else "(No life-event timeline provided.)")
    # )

    prompts = {}
    risk = risk_prompt(profile)
    prompts["G0"] = f"""/no_thinking {lang_tag}
                    {L['simulate']}
                    {L['use_profile']}
                    {basic_profile_text}
                    """
    # G1：人口信息学+大五人格+基础clinical
    prompts["G1"] = prompts["G0"] + personality_traits_text + risk


    prompts["G2"] = prompts["G1"] + f"2) {L['clinical']} {clinical_text} {risk}"

    # G2：人口信息学+大五人格+基础clinical+问诊对话信息
    clinical_messages = load_clinical_dialogues(profile_id, profile)
    prompts["G3"] = prompts["G2"] + f"3) {L['use_assessment_info']} {clinical_messages} {risk}"

    # G4：人口信息学+大五人格+基础clinical+咨询对话信息
    consultation_messages = load_consultation_dialogues(profile_id, profile)
    prompts["G4"] = prompts["G2"] + f"3) {L['use_consultation_info']} {consultation_messages} {risk}"

    # G5
    
    prompts["G5"] = prompts["G4"] + f"4) {L['use_assessment_info']} {clinical_messages} {risk}  "
    # G6
    if life_event_rendering_card:
        prompts["G6"] = prompts["G5"] + f"5) {L['use_event_cards']}{life_event_rendering_card}"
    else:
        prompts["G6"] = prompts["G5"]
        prompts["G6"] += f"""no timeline {risk}"""
    
    # G7

    prompts["G0.5"] = prompts["G0"] + f"0.5) {L['use_event_cards']}{life_event_rendering_card} {risk}"
    prompts["G1.5"] = prompts["G1"] + f"1.5) {L['use_event_cards']}{life_event_rendering_card}"
    prompts["G2.0"] = prompts["G1"] + f"2.0) {clinical_prompt_no_timeline(profile_template)} {risk}"
    prompts["G2.5"] = prompts["G2"] + f"2.5) {L['use_event_cards']}{life_event_rendering_card}"
    life_event_timeline = load_life_event_timeline(profile, 0, max_events_num = 50)
    prompts["G7"] = prompts["G5"] + f"5) {L['use_event_tl']}{life_event_timeline} {risk}"


    return prompts


def evaluate_patient_prompts(profile_template,
                             language="zh",
                             life_event_rendering_card = None,
                             symptom_rendering_card = None):
    """
    Generate system prompts for all baselines (G1~G6),
    with multi-language support ("zh" or "en").
    """
    profile = list(profile_template.values())[0]
    profile_id = list(profile_template.keys())[0]
    # 语言输出模板
    if language == "zh":
        lang_tag = "请你始终用中文回答问题。"
        L = {
            "use_profile": "请严格根据以下人口学信息来模拟患者：",
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
        - 口语化，整段不能超过100个汉字。

        """
        # - 只回答被问到的点；不要主动一次讲完全部背景。
        # - 不需要每次都表示理解/赞同/感谢；不确定就说“不知道/说不清”。
        L["use_consultation_info"] = """
        咨询参考：请根据以下咨询对话信息来模拟抑郁症患者进行心理咨询时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“说话方式”（语气、句子长短、犹豫、回避、表达情绪的方式）。
        - 你的回答以“来访者视角”说话，优先短句、口语化；允许停顿、含糊、说不清。


        【示例对话】
        """
        # - 不要每句都配合或总结咨询师的话；被追问时才多说一点点。

        L["use_assessment_info"] = """
        问诊参考：请根据以下问诊信息来模拟抑郁症患者与精神科医生对话时的说话方式和语言习惯：

        【few-shot使用规则】
        - 以下对话只示范“问诊场景的回答方式”：简短、对症状频率/程度的模糊描述、被追问再补充。
        - 只回答医生问到的点

        【示例对话】
        """

        #         - 只回答被问到的点；不要主动一次讲完全部背景。
        # - 不需要每次都表示理解/赞同/感谢；不确定就说“不知道/说不清”。

        L["clinical"] = """
        下面是临床信息，你会获得患者的阳性症状和阴性症状，以及临床总结。
        你需要在被问及阳性症状时表示肯定，在被问及阴性症状时否认。
        """



        L["use_event_cards"] = """
        请根据以下生活事件发帖，并体现事件触发/影响。
        要求：
        1) 只使用生活事件时间线中出现的事件与影响，不要编造具体原因/地点/人物。
        2) 你每次提到一个事件或其影响，必须明确说出对应时间点
        3) 若表达“从那以后/因为/导致”，必须让因果链落在生活事件时间线中（即事件与变化都来自生活事件时间线），否则不要写因果。
        4) 每次最多引用 1–2 个事件，避免堆砌， 尽量口语化，不要像在读列表。

       【生活事件时间线】
        """

        L["use_event_tl"] = """
        请根据以下【生活事件时间线卡片(cards)】表达情绪与功能变化，并体现事件触发/影响。
        要求：
        1) 只使用 cards 中出现的事件与影响，不要编造具体原因/地点/人物。
        2) 你每次提到一个事件或其影响，必须明确说出对应卡片的时间点
        3) 若表达“从那以后/因为/导致”，必须让因果链落在 cards 里（即事件与变化都来自 cards），否则不要写因果。
        4) 每次最多引用 1–2 张卡片，避免堆砌， 尽量口语化，不要像在读列表。

       【生活事件时间线卡片】
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



    # B prompt
    basic_profile_text = (
        f"{L['use_profile']}\n"
        f"Age: {profile['age']}\n"
        f"Gender: {profile['gender']}\n"
        f"Marital status: {profile['marital_status']}\n"
        f"Work status: {profile['work_status']}\n"
        )
    
    # P personality traits profile
    personality_traits_text = big_five_prompt(profile)
    

    # S1： symptom attributes
    positive_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["positive_symptoms"]])
    negative_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in profile["negative_symptoms"]])
    clinical_text = (
        (L["clinical"] + "\n")
        + "这些是患者出现的阳性症状:\n  - "
        + positive_symptoms_text
        + "\n\n这些是患者没有出现的症状:\n  - "
        + negative_symptoms_text
        + f"\n\n临床总结:\n{profile['summation']}"
    )

    clinical_text = clinical_prompt(profile_template) + f"\n\n临床总结:\n{profile['summation']}"

    # 时间线文本
    # event_timeline = load_life_event_timeline(profile, 0, max_events_num = 50)
    # event_tl_text = (
    #     f"{L['event_tl_label']}\n" +
    #     (event_timeline if event_timeline else "(No life-event timeline provided.)")
    # )

    prompts = {}
    risk = risk_prompt(profile)
    prompts["G0"] = f"""/no_thinking {lang_tag}
                    {L['simulate']}
                    {L['use_profile']}
                    {basic_profile_text}
                    """
    # G1：人口信息学+大五人格+基础clinical
    prompts["G1"] = prompts["G0"] + personality_traits_text + risk


    prompts["G2"] = prompts["G1"] + f"2) {L['clinical']} {clinical_text} {risk}"
    prompts["G1"] = prompts["G2"]
    # G2：人口信息学+大五人格+基础clinical+问诊对话信息
    clinical_messages = load_clinical_dialogues(profile_id, profile)
    prompts["G3"] = prompts["G2"] + f"3) {L['use_assessment_info']} {clinical_messages} {risk}"

    # G4：人口信息学+大五人格+基础clinical+咨询对话信息
    consultation_messages = load_consultation_dialogues(profile_id, profile)
    prompts["G4"] = prompts["G2"] + f"3) {L['use_consultation_info']} {consultation_messages} {risk}"

    # G5
    
    prompts["G5"] = prompts["G2"]
    # G6
    if life_event_rendering_card:
        prompts["G6"] = prompts["G2"]
    else:
        prompts["G6"] = prompts["G2"]
        prompts["G6"] += f"""no timeline {risk}"""
    
    # G7
    life_event_timeline = load_life_event_timeline(profile, 0, max_events_num = 50)
    prompts["G7"] = prompts["G2"]


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
