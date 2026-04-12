import json
import re
from typing import Any

def safe_json_loads(s: str) -> Any:
    """Parse JSON robustly (strip code fences / extract first {...})."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    s = s.strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    if not (s.startswith("{") and s.endswith("}")):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
    # print(s)
    return json.loads(s)


def days_to_relative_cn(days_ago: int) -> str:
    if days_ago <= 0:
        return "今天"
    if days_ago == 1:
        return "昨天"
    if days_ago < 7:
        return f"{days_ago}天前"
    if days_ago < 30:
        return f"{days_ago//7}周前"
    if days_ago < 365:
        return f"{days_ago//30}个月前"
    return f"{days_ago//365}年前"


def clamp_text(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= max_len else (s[:max_len] + "…")