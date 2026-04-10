def risk_prompt(profile: dict) -> str:
    """Safety / scope reminder appended to several baselines (research simulation)."""
    _ = profile
    return (
        "\n（研究用途角色扮演）请仅依据上文已给出的信息作答；不要编造未提供的诊断细节、"
        "具体病史或联系人信息；若信息不足请用口语表示说不清。\n"
    )
