# 评测 Prompt

- **`g_eval_main_template_zh.txt`**：G-Eval 主模板（多维度：真实度、人设、事件丰富度、症状一致性），与 `codes/evaluation/g_eval_new.py` 中 `get_eval_prompt` 一致。
- **`questions/`**：批量 QA 题集（`QA_ALL.json`、`QA_timeline.json` 等），与 `api.py` / `direct_api.py` 的 `--questions_path` 配合使用。

运行 G-Eval 前请设置 `OPENAI_API_KEY`（及可选 `OPENAI_BASE_URL`），并在 `g_eval_new.py` 中配置 `BASELINES_MAP`、`PROFILE_FILE` 等本地路径。
