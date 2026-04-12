import os
import json
from typing import Dict, Any, Literal, Optional

TimelineType = Literal["life_event", "symptom"]

TIMELINE_MEMORY_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "timelines"))




def load_render_cards(
    profile_id: str,
    tweet_user_id: str,
    timeline_type: TimelineType,
    candidate_index: int = 0,
    memory_root: str = TIMELINE_MEMORY_ROOT,
) -> Dict[str, Any]:
    """
    Load the JSON dict that contains {"cards":[...], ...}.
    """

    path = os.path.join(memory_root, timeline_type, f"{profile_id}_{tweet_user_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"render card file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def build_cards_prompt(
    profile_id: str,
    tweet_user_id: str,
    timeline_type: TimelineType,
    candidate_index: int = 0,
    max_cards: Optional[int] = None,  # None = all
    memory_root: str = TIMELINE_MEMORY_ROOT,
) -> str:
    """
    Minimal prompt block: rule + label + cards.
    """
    obj = load_render_cards(profile_id, tweet_user_id, timeline_type, candidate_index=candidate_index, memory_root=memory_root)
    cards = obj.get("cards", []) or []
    if max_cards is not None:
        cards = cards[:max_cards]

    rule = """
1. **单一时间嵌入**：提到事件/症状时，**只选用**卡片中的一个时间点（优先选“xx天前”或“xx周前”），像聊天一样自然地加在句子中间，**不要**使用括号或双重时间。
2. **口语化短句**：用连词（“也就是”、“打那之后”）把碎片信息串成一两句通顺的人话，禁止像清单一样罗列。
3. **严禁编造**：只说 Cards 里有的事，没有的细节不瞎编。

# 示例:
* [Card: 失眠 | 14天前]
  * ❌ 错误：失眠。14天前。
  * ❌ 错误：我有失眠症状（14天前）。
  * ✅ 正确：大概是**14天前**吧，我开始整晚整晚地睡不着觉。

* [Card: 离职 | 2个月前] -> [Card: 焦虑 | 2个月前]
  * ✅ 正确：自从**2个月前**我也离职了之后，心里就一直特别焦虑，根本静不下来。
    """
    label = "生活事件时间线卡片(cards)：" if timeline_type == "life_event" else "症状时间线卡片(cards)："

    lines = [rule, "", label]
    if not cards:
        lines.append("（无卡片）")
    else:
        for c in cards:
            # 每张卡单独一行：模型更容易引用
            lines.append(f"- {c.get('card_cn', '').strip()}")

    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    profile_id = "0003"
    with open(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "DEPROFILE", "deprofiles_complete_index.json")), "r") as f:
        profile_info = json.load(f)
    profile = profile_info.get(profile_id, {})
    tweet_user_id = "1220641557528756224"
    timeline_type = "life_event"
    candidate_index = 0
    max_cards = 10
    memory_root = TIMELINE_MEMORY_ROOT
    print(build_cards_prompt(profile_id, tweet_user_id, timeline_type, candidate_index, max_cards, memory_root))

    timeline_type = "symptom"
    print(build_cards_prompt(profile_id, tweet_user_id, timeline_type, candidate_index, max_cards, memory_root))