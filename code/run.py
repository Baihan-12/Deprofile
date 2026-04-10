# 5_app.py
import os
os.environ["SWIFT_DISABLE_PATCH"] = "1"
os.environ["SWIFT_PARALLEL_DISABLED"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import json
from utils import build_messages, generate_patient_prompts, load_clinical_dialogues


# ==========================================
# 2. 加载模型
# ==========================================
MODEL_DIR = os.getenv("DEPROFILE_MODEL_DIR", "Qwen/Qwen3-8B")


TEST_PROFILE = {
        "cr_id": "0",
        "d4_id": "971",
        "age": 19,
        "gender": "F",
        "marital_status": "single",
        "work_status": "student",
        "big_five": {
            "Openness": 6,
            "Conscientiousness": 5,
            "Extraversion": 3,
            "Agreeableness": 7,
            "Neuroticism": 2
        },
        "candidate_id": [
            {
                "basic_id": "837785700460265472",
                "similarity": 0.902693677739844,
                "symp_similarity": 1.0
            },
            {
                "basic_id": "173367742",
                "similarity": 0.8880416016036986,
                "symp_similarity": 1.0
            },
            {
                "basic_id": "1323237388211023872",
                "similarity": 0.807393770514232,
                "symp_similarity": 1.0
            },
            {
                "basic_id": "1249650669943955457",
                "similarity": 0.8070906378696021,
                "symp_similarity": 0.5
            }
        ],
        "positive_symptoms": [
            "任务-自杀-存在自杀倾向",
            "闲聊-自我表露-对事物的情绪",
            "任务-社会功能-避免从亲友处得到支持",
            "任务-躯体症状-躯体不适",
            "任务-精神状态-缺乏自信",
            "任务-兴趣-兴趣丧失超过两周",
            "闲聊-提供信息-主动提供相关信息",
            "任务-社会功能-学习工作存在困难",
            "任务-自杀-有无望感",
            "闲聊-自我表露-抱怨自我",
            "任务-躯体症状-运动性激越",
            "任务-精神状态-疲倦",
            "任务-兴趣-兴趣丧失",
            "任务-兴趣-范围-过去爱好"
        ],
        "negative_symptoms": [
            "任务-情绪-早晚差异",
            "任务-食欲-食欲存在问题",
            "任务-自杀-自我价值感低",
            "任务-情绪-情绪低落",
            "任务-睡眠-多梦",
            "任务-睡眠-存在睡眠问题",
            "任务-自杀-存在自杀行为"
        ],
        "summation": "来访者持续两周以上对事物缺乏兴趣、情绪烦躁、有无力感，另外来访者反映会有头痛，睡眠和食欲一般，有自杀的想法，会有无价值感。综上判断来访者抑郁程度为中度。因来访者反映情绪持续烦躁，需进一步进行双向障碍的排查。"
    }


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True
)
model.eval()


# ==========================================
# 3. 构造 chat 函数（现在用目标 baseline prompt）
# ==========================================



def chat_fn(profile_json_text, symptom_tl, event_tl, baseline_key, lang_choice, user_input, history):

    # 将 JSON profile 解析
    try:
        profile = json.loads(profile_json_text)
    except Exception:
        return history, "⚠️ Profile JSON 格式错误，请检查！"

    # 生成 prompts
    prompts = generate_patient_prompts(
        profile,
        symptom_timeline=symptom_tl,
        event_timeline=event_tl,
        language=lang_choice
    )

    system_prompt = prompts[baseline_key]

    messages = build_messages(system_prompt, "（系统已自动加载患者信息）", history, user_input)

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()

    history = history + [(user_input, reply)]
    return history, ""


# ==========================================
# 4. Gradio UI
# ==========================================
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 UPRP 模拟病人系统（带 baseline + 多语言选择）")

    with gr.Row():
        with gr.Column(scale=1):

            profile_json = gr.Textbox(
                label="Patient Profile (JSON)",
                lines=20,
                value=json.dumps(
                    TEST_PROFILE,
                    ensure_ascii=False,
                    indent=2
                )
            )

            symptom_tl = gr.Textbox(label="Symptom Timeline（可选）", lines=5)
            event_tl = gr.Textbox(label="Life Event Timeline（可选）", lines=5)

            baseline_key = gr.Dropdown(
                ["G1", "G2", "G3", "G4", "G5", "G6"],
                value="G6",
                label="选择 baseline 版本"
            )

            lang_choice = gr.Dropdown(
                ["zh", "en"],
                value="zh",
                label="语言（zh=中文, en=English）"
            )

        with gr.Column(scale=2):
            chat = gr.Chatbot(label="问诊对话（你是病人）", height=500)
            user_input = gr.Textbox(label="医生提问")
            clear_btn = gr.Button("清空")

    def clear_chat():
        return [], ""

    user_input.submit(
        chat_fn,
        inputs=[profile_json, symptom_tl, event_tl, baseline_key, lang_choice, user_input, chat],
        outputs=[chat, user_input]
    )

    clear_btn.click(clear_chat, None, [chat, user_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
