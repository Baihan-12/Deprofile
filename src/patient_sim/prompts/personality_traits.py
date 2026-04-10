def big_five_prompt(profile: dict) -> str:
    bf = profile.get("big_five") or {}
    if not bf:
        return ""
    bf_text = ", ".join([f"{k}: {v}" for k, v in bf.items()])
    return f"\n大五人格 (Big Five): {bf_text}\n"
