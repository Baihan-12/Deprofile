import json

def risk_prompt(profile):
    
    if profile["depression_risk"] == 0:
        depression_risk_text = "你目前没有抑郁症的风险。"
    elif profile["depression_risk"] == 1:
        depression_risk_text = "你目前有轻度抑郁症的风险。"
    elif profile["depression_risk"] == 2:
        depression_risk_text = "你目前有中度抑郁症的风险。"
    elif profile["depression_risk"] == 3:
        depression_risk_text = "你目前有重度抑郁症的风险。"
    if profile["suiside_risk"] == 0:
        suiside_risk_text = "你目前没有自杀的风险。"
    elif profile["suiside_risk"] == 1:
        suiside_risk_text = "你目前有轻度自杀的风险。"
    elif profile["suiside_risk"] == 2:
        suiside_risk_text = "你目前有中度自杀的风险。"
    elif profile["suiside_risk"] == 3:
        suiside_risk_text = "你目前有重度自杀的风险。"
    
    return depression_risk_text+suiside_risk_text+"尽管你可能有些症状和负面的生活事件，但请你记住你的抑郁症严重程度和自杀程度，如果你要填量表，请你根据这里给你的状况来填写。"