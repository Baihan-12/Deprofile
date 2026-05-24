import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import AzureOpenAI, OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = {
    'apiKey': os.getenv('OPENAI_API_KEY', ''),
    'model': os.getenv('OPENAI_MODEL', 'gemini-3-pro-preview'),
    'baseUrl': os.getenv('OPENAI_BASE_URL', 'https://aidp.bytedance.net/api/modelhub/online/v2/crawl'),
    'apiVersion': os.getenv('OPENAI_API_VERSION', '2024-02-01'),
    'apiType': os.getenv('OPENAI_API_TYPE', 'azure'),
}

DEMO_CONFIG = {
    'apiKey': os.getenv('DEPROFILE_DEMO_API_KEY', ''),
    'model': os.getenv('DEPROFILE_DEMO_MODEL', DEFAULT_CONFIG['model']),
    'baseUrl': os.getenv('DEPROFILE_DEMO_BASE_URL', DEFAULT_CONFIG['baseUrl']),
    'apiVersion': os.getenv('DEPROFILE_DEMO_API_VERSION', DEFAULT_CONFIG['apiVersion']),
    'apiType': os.getenv('DEPROFILE_DEMO_API_TYPE', DEFAULT_CONFIG['apiType']),
}
DEMO_MAX_TURNS = int(os.getenv('DEPROFILE_DEMO_MAX_TURNS', '20'))
DEMO_ENABLED = bool(DEMO_CONFIG['apiKey'])
DEMO_SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
DEMO_SESSION_USAGE: dict[str, dict[str, float | int]] = {}

SOURCE_FILES = {
    'selected_samples': REPO_ROOT / 'DEPROFILE' / 'selected_samples.json',
    'complete_index': REPO_ROOT / 'DEPROFILE' / 'deprofiles_complete_index.json',
}

SYMPTOM_LABEL_TRANSLATIONS = {
    '任务-兴趣-兴趣丧失': 'Loss of interest',
    '任务-兴趣-情感淡漠': 'Emotional blunting',
    '任务-兴趣-范围-所有事情': 'Loss of interest across all activities',
    '任务-兴趣-范围-过去爱好': 'Loss of interest in previous hobbies',
    '任务-情绪-情绪低落': 'Depressed mood',
    '任务-情绪-情绪低落超过两周': 'Depressed mood for more than two weeks',
    '任务-情绪-早晚差异': 'Diurnal mood variation',
    '任务-睡眠-入睡困难': 'Difficulty initiating sleep',
    '任务-睡眠-多梦': 'Vivid dreams',
    '任务-睡眠-存在睡眠问题': 'Sleep disturbance',
    '任务-睡眠-早醒': 'Early morning awakening',
    '任务-睡眠-睡眠时间少': 'Reduced sleep duration',
    '任务-睡眠-睡眠浅': 'Light sleep',
    '任务-社会功能-学习工作存在困难': 'Impaired study or work functioning',
    '任务-社会功能-日常生活存在困难': 'Difficulty with daily functioning',
    '任务-社会功能-避免与人接触': 'Social avoidance',
    '任务-社会功能-避免从亲友处得到支持': 'Avoids seeking support from family or friends',
    '任务-筛查-躁狂': 'Possible manic symptoms',
    '任务-筛查-遗传史': 'Family psychiatric history',
    '任务-精神状态-注意力不集中': 'Impaired concentration',
    '任务-精神状态-疲倦': 'Fatigue',
    '任务-精神状态-缺乏自信': 'Low self-confidence',
    '任务-精神状态-记忆力下降': 'Memory decline',
    '任务-精神状态-选择困难': 'Indecisiveness',
    '任务-自杀-存在自杀倾向': 'Suicidal tendency',
    '任务-自杀-存在自杀行为': 'History of suicidal behavior',
    '任务-自杀-存在自残倾向': 'Self-harm tendency',
    '任务-自杀-有无望感': 'Hopelessness',
    '任务-自杀-自我价值感低': 'Low self-worth',
    '任务-自杀-自罪': 'Self-blame or guilt',
    '任务-躯体症状-躯体不适': 'Somatic discomfort',
    '任务-躯体症状-运动性激越': 'Psychomotor agitation',
    '任务-躯体症状-运动性迟滞': 'Psychomotor retardation',
    '任务-食欲-显著体重变化': 'Marked weight change',
    '任务-食欲-暴饮暴食': 'Binge eating',
    '任务-食欲-食欲下降': 'Reduced appetite',
    '任务-食欲-食欲存在问题': 'Appetite disturbance',
    '闲聊-寻求帮助-询问医生的看法': "Asks for the clinician's opinion",
    '闲聊-提供信息-主动提供相关信息': 'Voluntarily provides relevant information',
    '闲聊-提供信息-被动提供相关信息': 'Passively provides relevant information',
    '闲聊-自我表露-对事物的情绪': 'Strong emotional reactions to events',
    '闲聊-自我表露-抱怨自我': 'Self-criticism',
}

SUMMARY_TRANSLATIONS = {
    '0069': 'The client reports depressed mood with diurnal variation, low self-worth, hopelessness, marked occupational impairment, social withdrawal, fatigue, irritability with psychomotor agitation, somatic dizziness, reduced self-confidence, guilt, prolonged sleep, binge eating with major weight gain, and suicidal thoughts without plan or attempt. Bipolar affective disorder is noted; specialist psychiatric evaluation is recommended.',
    '0091': 'The patient reports uncontrollable overeating despite prior dietary restriction, pronounced mood lability, difficulty initiating sleep, loss of interest in previously enjoyable activities, social impairment, and cognitive decline including poor memory. Bipolar affective disorder is indicated.',
    '0099': 'The patient presents with depressed mood, prominent somatic symptoms such as cold extremities, forehead heat, and mental clouding, diurnal variation with worse mornings, and occasional psychomotor slowing. Further psychological assessment is recommended; depressive state is indicated.',
    '0107': 'The patient presents with depressed mood, sleep disturbance with early awakening, reduced volition, low energy, indecisiveness, somatic symptoms, and low self-confidence, consistent with a depressive episode.',
    '0120': 'Over the past month, the client has shown low mood, loss of interest, and low energy, meeting core depressive criteria. Additional cognitive difficulty, sleep and appetite disturbance, and social dysfunction suggest a relatively severe depressive disorder; timely psychological or psychiatric care is recommended.',
    '0151': 'The client presents with depressed mood, reduced interest, hopelessness, low energy, poor sleep, impaired concentration, poor appetite, irritability, and somatic discomfort, alongside passive suicidal expressions. The client remains help-seeking; formal psychiatric evaluation and treatment are recommended.',
    '0559': 'There has been reduced interest for about two weeks, social withdrawal, fatigue, difficulty initiating sleep, feelings of worthlessness and guilt, with largely preserved social functioning. Findings are consistent with mild depression.',
    '0563': 'The client shows reduced interest, low self-evaluation, guilt, hopelessness, social avoidance, difficulty initiating sleep, irritability, and memory decline, with negative or suicidal cognitions. The client is willing to seek care; psychiatric evaluation and treatment are recommended.',
    '0767': 'The patient reports persistently depressed mood, inability to feel pleasure, worse symptoms in the morning and at bedtime, a clear precipitant of business failure, difficulty initiating sleep and early awakening with nightmares, irritability, restlessness, guilt about burdening the family, and low confidence in the future. A depressive episode is indicated, with bipolar disorder not ruled out.',
    '0770': 'Over the past month, the patient has had depressed mood with worse mornings, cognitive impairment and poor memory, low energy, occupational and daily-life impairment, self-blame and guilt, low self-esteem, suicidal thoughts, social withdrawal, hopelessness about the future, reduced activity, somatic discomfort, and irritability.',
    '0911': 'Depressed mood with worse mornings, reduced interest, sleep disturbance, reduced appetite, fatigue, and a two-week duration has impaired work and social functioning. Moderate depression is suggested; specialist evaluation, timely adjustment, exercise, and interpersonal support are recommended.',
    '1008': 'Over the past month, the patient reports depressed mood, reduced interest, social withdrawal, impairment in daily life, and low energy, consistent with a depressive episode.',
    '1100': 'Over the past month, the client reports reduced interest, low self-confidence, sleep disturbance, reduced appetite, weight loss, and psychomotor agitation or retardation, without suicidal ideation, and self-reports bipolar affective disorder. Specialist care and treatment adherence are recommended.',
    '1136': 'The client shows marked anhedonia, depressed mood, fatigue, reduced attention, slowed thinking, difficulty initiating sleep, binge eating, guilt, prior suicidal behavior, dizziness, irritability, and increased talkativeness. Both depressive and manic symptoms are present; specialist treatment and suicide intervention are recommended.',
    '1506': 'Over the past three months, the patient has experienced depressed mood, reduced interest, low energy, irritability with increased talkativeness, indecisiveness, cognitive impairment, social dysfunction, sleep disturbance with difficulty initiating sleep, low self-esteem, self-blame and guilt, hopelessness, and worse symptoms in the morning.',
    '1681': 'Over the past month, the client reports reduced interest, low mood especially in the morning, sleep disturbance with nightmares, fatigue, reduced self-confidence, binge eating with weight gain, occasional dizziness, social dysfunction, hopelessness, and a tendency toward self-harm, while remaining willing to seek help. Severe depressive tendency with mild suicide risk is suggested; early professional intervention is recommended.',
    '1795': 'Primary complaint is poor sleep with early awakening and daily fatigue. Social functioning is impaired, affecting daytime work and memory, and the patient is considering leave. Additional features include depressed mood, reduced interest, meaninglessness, occasional restlessness, and palpitations, with emotional eating but no current suicidal thoughts.',
    '1961': 'The patient reports depressed mood, inability to feel pleasure, generalized weakness and low energy, sleep disturbance with early awakening and nightmares, reduced interest in photography, anhedonia, social impairment, anxiety, and somatic discomfort, consistent with a depressive episode.',
    '2062': 'The patient reports persistent low mood, inability to feel happy, loss of interest, sleep disturbance with three hours less sleep than before, low self-confidence, feeling like a burden, guilt, social withdrawal, major occupational impairment, and cognitive decline, consistent with a depressive state.',
    '2310': 'There is low mood with morning worsening, reduced interest for more than two weeks, occupational impairment, fatigue, distractibility, memory decline, poorer decision-making, difficulty initiating sleep, binge eating, irritability, psychomotor slowing, dizziness, sweating, psychomotor agitation, and self-harm thoughts without action.',
    '2556': 'Over the past month, the client has shown low mood, loss of interest, and reduced energy, meeting core depressive criteria. Additional reduced attention, sleep and appetite disturbance, low self-evaluation, and suicidal thoughts indicate severe depression with mild suicide risk; timely psychological or psychiatric care is recommended.',
    '2599': 'The patient presents with depressed mood, reduced interest, hopelessness and helplessness, worse symptoms in the morning, anxiety, and rapid speech, with impairment in work and daily life, consistent with a depressive state.',
    '2652': 'Over the past month, the client has shown low mood, low energy, and loss of interest, meeting core depressive criteria. Additional reduced attention, low self-evaluation, and sleep and appetite disturbance indicate severe depression; timely psychological or psychiatric care is recommended.',
    '2737': 'Over the past month, the patient has had depressed mood, reduced interest, low energy, self-blame and guilt, hopelessness and helplessness, self-harm thoughts, sleep disturbance with light sleep and early awakening, psychomotor slowing, reduced volition and behavior, and social withdrawal.',
    '2798': 'Depressed mood with worse mornings, anhedonia, low energy, reduced attention, guilt, worthlessness, poorer sleep, appetite change, psychomotor slowing, and suicidal thoughts with a prior suicide attempt have persisted for one month and impaired normal study and daily life. Moderate depression with suicidal tendency is suggested; urgent specialist care, treatment adherence, social support, and adaptive coping are recommended.',
    '2805': 'Depressed mood worsens at night, with anhedonia lasting three weeks, impaired personal hygiene, reduced work efficiency, social dysfunction, low self-confidence, worthlessness, poorer decision-making, reduced sleep, reduced appetite, irritability, dizziness, dyspnea, and suicidal thoughts without action. No family history is reported.',
    '2960': 'Over the past two weeks, the client has shown reduced confidence, occupational dysfunction, depressed mood with stronger morning and evening symptoms, reduced interest, weight gain, slowed responses, suicidal ideation, guilt, social instability, psychomotor agitation, hopelessness, and low self-worth. Symptoms are consistent with moderately severe depression with suicide risk; specialist evaluation and treatment are recommended.',
}


def _read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _prune_demo_sessions() -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, entry in DEMO_SESSION_USAGE.items()
        if now - float(entry.get('updatedAt', now)) > DEMO_SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        DEMO_SESSION_USAGE.pop(session_id, None)


def _get_demo_turns_used(session_id: str) -> int:
    if not session_id:
        return 0
    _prune_demo_sessions()
    entry = DEMO_SESSION_USAGE.get(session_id, {})
    return int(entry.get('usedTurns', 0))


def get_demo_status(session_id: str = '') -> dict[str, Any]:
    used_turns = _get_demo_turns_used(session_id)
    remaining_turns = max(DEMO_MAX_TURNS - used_turns, 0) if DEMO_ENABLED else 0
    return {
        'enabled': DEMO_ENABLED,
        'maxTurns': DEMO_MAX_TURNS,
        'usedTurns': used_turns,
        'remainingTurns': remaining_turns,
    }


def _consume_demo_turn(session_id: str) -> dict[str, Any]:
    if not session_id:
        raise PermissionError('Demo mode requires a browser session ID.')
    status = get_demo_status(session_id)
    if not status['enabled']:
        return status
    if status['remainingTurns'] <= 0:
        raise PermissionError('The built-in demo key has reached its 20-turn limit for this browser session.')
    DEMO_SESSION_USAGE[session_id] = {
        'usedTurns': int(status['usedTurns']) + 1,
        'updatedAt': time.time(),
    }
    return get_demo_status(session_id)


def _resolve_effective_config(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if config.get('apiKey'):
        return config, False
    if DEMO_ENABLED:
        return DEMO_CONFIG, True
    return config, False


@lru_cache(maxsize=2)
def load_profiles(source: str) -> dict[str, dict[str, Any]]:
    path = SOURCE_FILES[source]
    data = _read_json(path)
    return {str(pair_id): profile for pair_id, profile in data.items()}


def _risk_level_text(value: Any) -> str:
    mapping = {0: 'Low', 1: 'Mild', 2: 'Moderate', 3: 'High'}
    try:
        return mapping[int(value)]
    except Exception:
        return str(value)


def _translate_symptom_label(label: str) -> str:
    return SYMPTOM_LABEL_TRANSLATIONS.get(label, label.split('-')[-1].replace('_', ' ').strip())


def _translate_summary(pair_id: str, summary: str) -> str:
    if pair_id in SUMMARY_TRANSLATIONS:
        return SUMMARY_TRANSLATIONS[pair_id]
    return summary


def _normalize_profile(pair_id: str, profile: dict[str, Any]) -> dict[str, Any]:
    return {
        'pairId': pair_id,
        'age': profile.get('age', 'Unknown'),
        'gender': profile.get('gender', 'Unknown'),
        'maritalStatus': profile.get('marital_status', 'Unknown'),
        'workStatus': profile.get('work_status', 'Unknown'),
        'depressionRisk': profile.get('depression_risk', 'Unknown'),
        'depressionRiskLabel': _risk_level_text(profile.get('depression_risk', 'Unknown')),
        'suicideRisk': profile.get('suiside_risk', 'Unknown'),
        'suicideRiskLabel': _risk_level_text(profile.get('suiside_risk', 'Unknown')),
        'candidateCount': len(profile.get('candidate_id', [])),
        'crId': profile.get('cr_id', ''),
        'd4Id': profile.get('d4_id', ''),
        'bigFive': profile.get('big_five', {}),
        'positiveSymptoms': [_translate_symptom_label(item) for item in profile.get('positive_symptoms', [])],
        'negativeSymptoms': [_translate_symptom_label(item) for item in profile.get('negative_symptoms', [])],
        'summary': _translate_summary(pair_id, profile.get('summation', '')),
    }


def list_profile_summaries(source: str) -> list[dict[str, Any]]:
    profiles = load_profiles(source)
    items = [_normalize_profile(pair_id, profile) for pair_id, profile in profiles.items()]
    items.sort(key=lambda item: item['pairId'])
    return items


def _timeline_path(profile: dict[str, Any], timeline_type: str) -> Path | None:
    candidates = profile.get('candidate_id') or []
    if not candidates:
        return None
    basic_id = str(candidates[0].get('basic_id', '')).strip()
    if not basic_id:
        return None
    return REPO_ROOT / 'timeline' / f'stmhd_{timeline_type}_timeline' / f'{basic_id}.json'


def _load_life_event_timeline(profile: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    path = _timeline_path(profile, 'life_event')
    if path is None or not path.exists():
        return []
    payload = _read_json(path)
    timeline = payload.get('timeline', [])
    return timeline[-limit:]


def build_timeline_preview(profile: dict[str, Any], limit: int = 6) -> list[str]:
    preview = []
    for item in _load_life_event_timeline(profile, limit=limit)[::-1]:
        timestamp = item.get('timestamp', '?')
        event_text = item.get('life_event', '')
        tweet_text = item.get('tweet', '')
        preview.append(f"T-{timestamp}d | {event_text} | {tweet_text}")
    return preview


def _score_band(score: int) -> str:
    if score <= 2:
        return 'low'
    if score >= 6:
        return 'high'
    return 'moderate'


def _big_five_lines(profile: dict[str, Any]) -> str:
    descriptors = {
        'Openness': {
            'high': 'curious, abstract, and imaginative',
            'moderate': 'balanced between practicality and curiosity',
            'low': 'concrete, cautious, and conventional',
        },
        'Conscientiousness': {
            'high': 'structured, disciplined, and detail-oriented',
            'moderate': 'reliable with some flexibility',
            'low': 'disorganized, loose, and less planful',
        },
        'Extraversion': {
            'high': 'energetic, expressive, and socially forward',
            'moderate': 'socially comfortable but not dominant',
            'low': 'reserved, brief, and inward-facing',
        },
        'Agreeableness': {
            'high': 'gentle, accommodating, and conflict-avoidant',
            'moderate': 'friendly but still able to hold boundaries',
            'low': 'blunt, skeptical, and less emotionally accommodating',
        },
        'Neuroticism': {
            'high': 'emotionally reactive, worried, and sensitive to stress',
            'moderate': 'emotionally responsive but usually recoverable',
            'low': 'calm, steady, and hard to destabilize',
        },
    }
    entries = []
    for trait, score in profile.get('big_five', {}).items():
        band = _score_band(int(score))
        entries.append(f"- {trait}: score {score}; present as {descriptors.get(trait, {}).get(band, band)}")
    return '\n'.join(entries)


def _symptom_list(items: list[str]) -> str:
    cleaned = [f"- {item}" for item in items if item]
    return '\n'.join(cleaned) if cleaned else '- None provided'


def build_system_prompt(source: str, pair_id: str, language: str = 'en') -> str:
    profiles = load_profiles(source)
    profile = profiles[pair_id]
    normalized = _normalize_profile(pair_id, profile)
    timeline_items = _load_life_event_timeline(profile, limit=8)
    timeline_text = '\n'.join(
        f"- {item.get('timestamp', '?')} days before present: {item.get('life_event', '')}. Evidence tweet: {item.get('tweet', '')}"
        for item in timeline_items
    ) or '- No life-event timeline available.'

    return f"""You are simulating a mental-health patient for research and demonstration.
Respond in natural English only.
Stay in first person.
Do not act like an assistant, clinician, or evaluator.
Answer only what is asked.
Keep responses concise, conversational, and emotionally grounded.
Do not invent facts, events, symptoms, or relationships that are not supported by the profile below.
If the user asks about timing, use the timeline evidence when available.
If something is uncertain or unsupported, say you are not sure instead of fabricating.

Patient profile
- Pair ID: {normalized['pairId']}
- Age: {normalized['age']}
- Gender: {normalized['gender']}
- Marital status: {normalized['maritalStatus']}
- Work status: {normalized['workStatus']}
- Depression risk: {normalized['depressionRiskLabel']}
- Suicide risk: {normalized['suicideRiskLabel']}

Big Five guidance
{_big_five_lines(profile)}

Positive symptoms to acknowledge when relevant
{_symptom_list(normalized['positiveSymptoms'])}

Negative symptoms to deny when relevant
{_symptom_list(normalized['negativeSymptoms'])}

Clinical summary
{normalized['summary'] or 'No clinical summary provided.'}

Dialogue behavior
- Sound like a patient in a psychiatric or counseling conversation.
- Use short to medium-length answers unless the question clearly asks for detail.
- Avoid bullet lists in normal dialogue.
- Do not summarize the whole case unless explicitly asked.
- Preserve internal consistency across turns.

Life-event timeline
{timeline_text}
"""


def get_profile_detail(source: str, pair_id: str, language: str = 'en') -> dict[str, Any]:
    profiles = load_profiles(source)
    if pair_id not in profiles:
        raise KeyError(pair_id)
    profile = profiles[pair_id]
    return {
        'profile': _normalize_profile(pair_id, profile),
        'timelinePreview': build_timeline_preview(profile),
        'systemPrompt': build_system_prompt(source, pair_id, language=language),
    }


def validate_config(config: dict[str, Any], session_id: str = '') -> dict[str, Any]:
    missing = []
    if not config.get('model'):
        missing.append('OPENAI_MODEL')
    if not config.get('baseUrl'):
        missing.append('OPENAI_BASE_URL')
    if not config.get('apiKey') and not DEMO_ENABLED:
        missing.append('OPENAI_API_KEY')
    if missing:
        return {'ok': False, 'message': f"Missing required config: {', '.join(missing)}"}
    if config.get('apiKey'):
        return {'ok': True, 'message': 'Configuration looks complete. Your custom API key will be used.'}
    if DEMO_ENABLED:
        status = get_demo_status(session_id)
        return {
            'ok': True,
            'message': f"Anonymous demo mode is enabled. {status['remainingTurns']} of {status['maxTurns']} turns remain in this browser session.",
        }
    return {'ok': True, 'message': 'Configuration looks complete. You can start chatting.'}


def _build_clients(config: dict[str, Any]) -> list[tuple[str, Any]]:
    api_key = config.get('apiKey', '')
    base_url = (config.get('baseUrl') or '').rstrip('/')
    api_version = config.get('apiVersion') or '2024-02-01'
    api_type = (config.get('apiType') or '').lower()
    clients: list[tuple[str, Any]] = []

    if api_type == 'azure':
        clients.append(('azure', AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=base_url)))
        clients.append(('openai-compatible', OpenAI(api_key=api_key, base_url=base_url)))
    else:
        clients.append(('openai-compatible', OpenAI(api_key=api_key, base_url=base_url)))
        if api_version:
            clients.append(('azure-fallback', AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=base_url)))
    return clients


def _create_completion(config: dict[str, Any], messages: list[dict[str, str]]) -> Any:
    model = config.get('model', DEFAULT_CONFIG['model'])
    errors: list[str] = []
    for name, client in _build_clients(config):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.8,
                max_tokens=900,
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError('Model request failed. ' + ' | '.join(errors))


def chat_with_profile(
    config: dict[str, Any],
    source: str,
    pair_id: str,
    messages: list[dict[str, str]],
    language: str = 'en',
    session_id: str = '',
) -> dict[str, Any]:
    detail = get_profile_detail(source, pair_id, language=language)
    chat_messages = [{'role': 'system', 'content': detail['systemPrompt']}]
    chat_messages.extend(messages)
    effective_config, using_demo_key = _resolve_effective_config(config)

    if using_demo_key and get_demo_status(session_id)['remainingTurns'] <= 0:
        raise PermissionError('The built-in demo key has reached its 20-turn limit for this browser session.')

    response = _create_completion(effective_config, chat_messages)
    reply = response.choices[0].message.content or ''
    finish_reason = getattr(response.choices[0], 'finish_reason', None)

    if finish_reason == 'length' and reply.strip():
        continuation_messages = [
            *chat_messages,
            {'role': 'assistant', 'content': reply.strip()},
            {
                'role': 'user',
                'content': 'Continue the same answer from exactly where you stopped. Do not repeat previous sentences.',
            },
        ]
        continuation = _create_completion(effective_config, continuation_messages)
        extra = continuation.choices[0].message.content or ''
        if extra.strip():
            reply = f"{reply.rstrip()} {extra.lstrip()}".strip()

    demo_status = get_demo_status(session_id)
    if using_demo_key:
        demo_status = _consume_demo_turn(session_id)

    return {
        'reply': reply.strip(),
        'systemPrompt': detail['systemPrompt'],
        'profile': detail['profile'],
        'timelinePreview': detail['timelinePreview'],
        'demoStatus': demo_status,
        'usedDemoKey': using_demo_key,
    }
