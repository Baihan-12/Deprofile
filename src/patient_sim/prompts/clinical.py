from __future__ import annotations

from typing import Any, Dict, Union

ProfileDict = Dict[str, Any]
ProfileTemplate = Union[ProfileDict, Dict[str, ProfileDict]]


def _resolve_profile(profile_or_template: ProfileTemplate) -> ProfileDict:
    if not isinstance(profile_or_template, dict):
        raise TypeError("profile must be a dict")
    if "positive_symptoms" in profile_or_template or (
        "summation" in profile_or_template and "candidate_id" in profile_or_template
    ):
        return profile_or_template
    if profile_or_template and all(isinstance(v, dict) for v in profile_or_template.values()):
        return next(iter(profile_or_template.values()))
    raise ValueError("clinical_prompt: expected a profile dict or a mapping {id: profile}")


_CLINICAL_HEADER = """
下面是临床信息，你会获得患者的阳性症状和阴性症状。
你需要在被问及阳性症状时表示肯定，在被问及阴性症状时否认。
""".strip()


def clinical_prompt(profile_or_template: ProfileTemplate) -> str:
    profile = _resolve_profile(profile_or_template)
    pos = profile.get("positive_symptoms") or []
    neg = profile.get("negative_symptoms") or []
    positive_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in pos])
    negative_symptoms_text = "\n  - ".join([sym.split("-")[-1] for sym in neg])
    return (
        _CLINICAL_HEADER
        + "\n这些是患者出现的阳性症状，请在被问到时肯定这些症状:\n  - "
        + positive_symptoms_text
        + "\n\n这些是患者没有出现的症状，请在被问到时否认这些症状:\n  - "
        + negative_symptoms_text
    )


def clinical_prompt_no_timeline(profile_template: ProfileTemplate) -> str:
    """Ablated clinical block: discourage inferring symptom trajectories from time."""
    base = clinical_prompt(profile_template)
    return base + "\n（注意：请勿引用或推断症状随时间的变化；仅依据上述静态症状列表回答。）\n"
