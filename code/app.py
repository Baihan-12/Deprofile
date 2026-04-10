# app.py
import gradio as gr
import requests

FASTAPI_BASE = "http://127.0.0.1:9000"


# -----------------------------
# 开始问诊：生成 user_name + profile
# -----------------------------
def start_session_fn(user_name):
    if not user_name:
        return "⚠️ 请先输入昵称！", None, None, []

    # 请求后端随机抽 profile
    resp = requests.get(f"{FASTAPI_BASE}/api/random_profile")
    data = resp.json()

    profile_id = data["profile_id"]
    profile_info = data["profile_info"]

    display_text = f"""
### 🧬 本次问诊模拟病人：{profile_id}

**年龄**：{profile_info['age']}  
**性别**：{profile_info['gender']}  
**职业**：{profile_info['work_status']}  
**婚姻状态**：{profile_info['marital_status']}  
**大五人格**：{profile_info['big_five']}  
"""

    # 返回显示内容 + 保存的状态
    return display_text, user_name, profile_id, []


# -----------------------------
# 聊天
# -----------------------------
def chat_fn(user_input, history, baseline, lang, session_profile_id):
    if not session_profile_id:
        return history, "⚠️ 请先点击“开始问诊”。"

    payload = {
        "history": history,
        "query": user_input,
        "baseline": baseline,
        "language": lang,
        "profile_id": session_profile_id
    }

    try:
        resp = requests.post(f"{FASTAPI_BASE}/api/chat", json=payload)
        data = resp.json()
        return data["history"], ""
    except Exception as e:
        return history, f"❌ FastAPI 服务连接失败: {e}"


# -----------------------------
# 提交 session
# -----------------------------
def submit_session_fn(history, realism, persona, symptom_fit, event_fit, comment,
                      session_user_name, session_profile_id):

    if not session_user_name or not session_profile_id:
        return "⚠️ 请先点击“开始问诊”。"

    if not history:
        return "⚠️ 没有可提交的对话。"

    # 评分是否填写
    if realism is None and persona is None and symptom_fit is None and event_fit is None and not comment:
        evaluation = None
    else:
        evaluation = {
            "realism": realism,
            "persona": persona,
            "symptom_fit": symptom_fit,
            "event_fit": event_fit,
            "comment": comment
        }

    payload = {
        "user_name": session_user_name,
        "profile_id": session_profile_id,
        "chat_history": history,
        "evaluation": evaluation
    }

    try:
        resp = requests.post(f"{FASTAPI_BASE}/api/submit_session", json=payload)
        return f"✅ 已保存到文件：{resp.json()['file']}"
    except Exception as e:
        return f"❌ 提交失败: {e}"


# ======================================================
# 构建 Gradio UI（所有组件必须在 Blocks 内部）
# ======================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 模拟病人问诊系统（含可选评测 + 病人加载）")

    # 状态变量
    session_user_name = gr.State()
    session_profile_id = gr.State()

    # -----------------------------
    # 开始问诊区
    # -----------------------------
    with gr.Accordion("🩺 开始新的问诊", open=True):
        user_name_input = gr.Textbox(label="请输入您的昵称")
        start_btn = gr.Button("开始问诊")
        session_profile_info = gr.Markdown("这里将显示问诊病人的基本信息")

    # 聊天窗口（此时 chat 已存在，能 safely 被 start_session_fn 输出重置）
    chat = gr.Chatbot(label="对话窗口", height=500)

    start_btn.click(
        start_session_fn,
        inputs=[user_name_input],
        outputs=[session_profile_info, session_user_name, session_profile_id, chat]
    )

    # -----------------------------
    # 聊天部分
    # -----------------------------
    with gr.Row():
        baseline = gr.Dropdown(["G1", "G2", "G3", "G4", "G5", "G6"],
                               value="G1", label="选择 baseline 版本")
        lang = gr.Dropdown(["zh", "en"], value="zh", label="对话语言")

    user_input = gr.Textbox(label="医生提问")
    clear_btn = gr.Button("清空对话")

    user_input.submit(
        chat_fn,
        inputs=[user_input, chat, baseline, lang, session_profile_id],
        outputs=[chat, user_input]
    )

    clear_btn.click(lambda: ([], ""), None, [chat, user_input])

    # -----------------------------
    # 评测部分
    # -----------------------------
    with gr.Accordion("⭐ 对话评测（可选，不填也能提交）", open=False):
        realism = gr.Slider(0, 5, step=1, label="真实性（像真实病人）")
        persona = gr.Slider(0, 5, step=1, label="符合人设程度")
        symptom_fit = gr.Slider(0, 5, step=1, label="符合症状程度")
        event_fit = gr.Slider(0, 5, step=1, label="生活事件真实性与合理性")
        comment = gr.Textbox(label="自由点评（可选）", lines=3)

    submit_btn = gr.Button("📤 提交对话与（可选）评分")
    submit_status = gr.Markdown()

    submit_btn.click(
        submit_session_fn,
        inputs=[chat, realism, persona, symptom_fit, event_fit, comment,
                session_user_name, session_profile_id],
        outputs=[submit_status]
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    debug=True
)
