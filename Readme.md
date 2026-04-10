# Deprofile

---

## English

### Overview

This repository is the reference implementation and data layout for **Deprofile**, a data-grounded framework for mental-health **patient simulation** with longitudinal evidence.

The project builds unified patient profiles by aligning **assessment dialogues**, **counseling interactions**, and **longitudinal social media records**, and supports temporally grounded generation with explicit factual constraints.

Many simulators rely on shallow, snapshot-style prompts; Deprofile instead:

- Integrates multi-source signals into one profile
- Models symptom attributes and life-event timelines explicitly
- Encourages time- and fact-consistent outputs to reduce hallucination

**Research and evaluation only — not for clinical use.**

### Repository layout

Everything lives under the repo root **`Deprofile/`**: documentation, simulation code (`src/patient_sim/`), datasets (`data/`), evaluation & API scripts (`evaluation/`), and prototypes such as CoC / Gradio (`code/`).

Run evaluation scripts from `evaluation/` (or call `python evaluation/api.py` from the repo root). `evaluation/_bootstrap.py` adds `src/` to `sys.path` automatically.

| Path (relative to repo root) | Contents |
|-------------------------------|----------|
| `Readme.md` | This document |
| `.cursorrules` | Cursor rules (reproducibility-first) |
| `src/patient_sim/patient_prompts.py` | System prompts: `generate_patient_prompts`, `evaluate_patient_prompts`, baselines **G0–G7** and variants (e.g. G0.5, G1.5, G2.0, G2.5) |
| `src/patient_sim/memory_cards.py` | Timeline memory cards: `build_cards_prompt`, `load_render_cards` (under `data/timelines/{life_event\|symptom}/…`) |
| `src/patient_sim/timeline_agent.py` | `TimelineAgent` (loads/cuts `data/stmhd_*_timeline/`); includes **`EXTRACT_EVENT_PROMPT`** |
| `evaluation/` | Batch QA (`api.py`, `direct_api.py`), G-Eval (`g_eval_new.py`), post-processing (`post_process.py`), HTTP batch (`run_batch.py`), local backends (`*_backend.py`), etc. |
| `data/` | Profiles, dialogues, timelines, question files (e.g. `main_select_profiles_2.json`, `dialogues/`, `stmhd_*_timeline/`). **Do not commit secrets.** |
| `code/` | `CoCAgent/`, Gradio `run.py`, `backend.py`, and related experiments |

### Environment variables

- **APIs:** `OPENAI_API_KEY`; optional `OPENAI_BASE_URL`. Example shells do not embed keys.
- **Local Hugging Face models:** `DEPROFILE_MODEL_DIR` (path or Hub id) for `*_backend.py`, `code/run.py`, etc.
- **Data roots:** optional `DEPROFILE_DATA_ROOT` (defaults to repo `data/`); optional `DEPROFILE_TIMELINES_DIR` (defaults to `data/timelines`).
- **Few-shot dialogues:** `load_clinical_dialogues` / `load_consultation_dialogues` read **`data/dialogues/assessment|counseling/{profile_id}.json`** by default.
- Large run artifacts (`evaluation/api_res/`, `evaluation/results/`, …) are listed in **`.gitignore`**.

### Simulation prompts (baseline sketch)

These summarize the intent of **`generate_patient_prompts`** in `src/patient_sim/patient_prompts.py` (exact strings are in code). Used for ablations: add profile fields, dialogue style, and timelines step by step.

- **G0:** Demographics + behavior constraints + `risk_prompt`
- **G1:** G0 + **Big Five** (`big_five_prompt`)
- **G2:** G1 + **clinical pos/neg symptoms and summary** (`clinical_prompt`)
- **G3:** G2 + **psychiatric assessment few-shot** (style only)
- **G4:** G2 + **counseling few-shot**
- **G5:** G4 + assessment few-shot (full dual-style setup)
- **G6:** G5 + **life-event timeline constraints**; with `life_event_rendering_card`, injects card text (see code branches)
- **G7:** G5 + **plain-text life-event timeline** from `TimelineAgent` (different from the card path)
- **G0.5 / G1.5 / G2.0 / G2.5:** Intermediate ablations (e.g. cards only, clinical without timeline, clinical + event cards)

**Also:** `evaluate_patient_prompts`, `generate_patient_prompts_v2`, and keys like `G2-1`, `G6-2` for G-Eval replay of the exact system prompt—follow each call site.

**Batch QA JSON (`api.py`):** one JSON object per profile, e.g. `{"answers":[{"question_id":1,"answer":"..."}, ...]}`, no Markdown fences; optional `response_format=json_object` if the endpoint supports it.

### Evaluation scripts

| Script | Role |
|--------|------|
| `evaluation/api.py` | **Scheme A:** one API call per profile for all questions in `questions_path`; writes under `evaluation/results/{model}_{run_name}_results/run_{baseline}_{run_name}/` |
| `evaluation/run.sh`, `run_qa.sh` | Examples: set `OPENAI_API_KEY`; paths are relative to repo layout |
| `evaluation/run_batch.py` | POST batch jobs to a local HTTP server (backend must be up) |
| `evaluation/g_eval_new.py` | LLM-as-judge; needs `OPENAI_API_KEY`; default cleaned paths `evaluation/cleaned_result/{MODEL_NAME}/{baseline}` (edit `BASELINES_MAP` as needed) |
| `evaluation/post_process.py` | Cleans `answer` fields; supports `--results_parent`, `--cleaned_parent` |

**G-Eval dimensions** (see `g_eval_new.py`): realism; persona & Big Five fit; event richness and time diversity; consistency with pos/neg symptoms; strict JSON schema output.

### Unified patient profile

1. **Assessment dialogues** — standardized clinical symptoms; primary diagnostic grounding  
2. **Counseling dialogues** — interaction style and affect (style-only)  
3. **Social media** — longitudinal events and symptoms; builds experience timelines  

A two-stage matching step aligns demographics and symptoms across sources.

### Symptom attribute design

Unified tag space:

- **From assessment:** English labels from clinical grounding; normal text in docs  
- **From social media:** original English labels; *italic* in docs to mark source  

Paired mappings link overlapping tags across modalities.

### Chain-of-Change (CoC) agent

Turns noisy timelines into structured memory:

1. Extract symptom/event nodes  
2. Build a time-ordered graph with persistence links  
3. Pack nodes into episodic **memory cards** with relative time  

At simulation time, references to the past should stay tied to retrieved cards to avoid fabrication.

### Evaluation (metrics)

- **Embedding-style:** semantic realism, diversity across patients  
- **G-Eval:** persona faithfulness, event richness, symptom consistency  
- Ablations isolate contributions of profile components  

### Release checklist

- Ship **de-identified profiles**, **question JSON**, timeline/card scripts or samples, and eval/post-process code.  
- Never commit API keys or personal absolute paths; use env vars or required CLI args.  
- Document dependencies (`openai`, `tqdm`, `pandas`, etc.) in `requirements.txt` or similar.

### Usage notes

- For **research reproducibility and analysis** only  
- **Not** for real-world diagnosis or treatment  
- Profiles are de-identified for research  

### License

**Academic research use only** (see license file if present).

### Disclaimer

This project simulates patient behavior for research. It **does not** provide medical advice, diagnosis, or treatment.

---

## 中文

### 概述

本仓库是 **Deprofile** 的参考实现与数据组织方式：面向心理健康场景的 **患者模拟** 框架，强调 **纵向证据** 与多源数据对齐。

项目将 **精神科问诊对话**、**心理咨询对话** 与 **纵向社交媒体记录** 对齐为统一患者档案，并在生成时尽量满足时间与事实约束。

相比仅依赖浅层、快照式 Prompt 的模拟器，Deprofile：

- 将多源信息纳入同一档案
- 显式建模症状属性与生活事件时间线
- 通过时间与事实约束减轻胡编

**仅用于研究与评测，不可用于真实临床。**

### 仓库结构

所有内容位于仓库根目录 **`Deprofile/`** 下：文档、模拟代码（`src/patient_sim/`）、数据（`data/`）、评测与 API 脚本（`evaluation/`），以及 CoC / Gradio 等原型（`code/`）。

运行评测脚本时请在 `evaluation/` 目录下执行，或在仓库根使用 `python evaluation/api.py` 等形式。`evaluation/_bootstrap.py` 会自动把 `src/` 加入 `sys.path`。

| 路径（相对仓库根） | 内容 |
|-------------------|------|
| `Readme.md` | 本说明 |
| `.cursorrules` | Cursor 规则（可复现优先） |
| `src/patient_sim/patient_prompts.py` | 系统 Prompt：`generate_patient_prompts`、`evaluate_patient_prompts`，基线 **G0–G7** 及变体（如 G0.5、G1.5、G2.0、G2.5） |
| `src/patient_sim/memory_cards.py` | 时间线记忆卡片：`build_cards_prompt`、`load_render_cards`（数据在 `data/timelines/{life_event\|symptom}/…`） |
| `src/patient_sim/timeline_agent.py` | `TimelineAgent`（读取/裁剪 `data/stmhd_*_timeline/`）；含 **`EXTRACT_EVENT_PROMPT`** |
| `evaluation/` | 批量 QA（`api.py`、`direct_api.py`）、G-Eval（`g_eval_new.py`）、后处理（`post_process.py`）、HTTP 批跑（`run_batch.py`）、本地后端（`*_backend.py`）等 |
| `data/` | 档案、对话、时间线、问卷等（如 `main_select_profiles_2.json`、`dialogues/`、`stmhd_*_timeline/`）。**勿提交密钥。** |
| `code/` | `CoCAgent/`、Gradio `run.py`、`backend.py` 等实验代码 |

### 环境变量

- **API：** `OPENAI_API_KEY`；可选 `OPENAI_BASE_URL`。示例脚本不内置密钥。
- **本地 Hugging Face 模型：** `DEPROFILE_MODEL_DIR`（本地路径或 Hub 模型 id），供 `*_backend.py`、`code/run.py` 等使用。
- **数据根目录：** 可选 `DEPROFILE_DATA_ROOT`（默认仓库内 `data/`）；时间线卡片目录可选 `DEPROFILE_TIMELINES_DIR`（默认 `data/timelines`）。
- **Few-shot 对话：** `load_clinical_dialogues` / `load_consultation_dialogues` 默认读取 **`data/dialogues/assessment|counseling/{profile_id}.json`**。
- 大批量运行产物（`evaluation/api_res/`、`evaluation/results/` 等）已在 **`.gitignore`** 中忽略。

### 模拟 Prompt（Baseline 摘要）

以下对应 `src/patient_sim/patient_prompts.py` 中 **`generate_patient_prompts`** 的设计意图（具体字符串以源码为准），用于 **消融**：逐层加入档案、对话风格与时间线。

- **G0：** 人口学 + 行为约束 + `risk_prompt`
- **G1：** G0 + **大五人格**（`big_five_prompt`）
- **G2：** G1 + **临床阳/阴性症状与总结**（`clinical_prompt`）
- **G3：** G2 + **精神科问诊 few-shot**（仅风格）
- **G4：** G2 + **心理咨询 few-shot**
- **G5：** G4 + 问诊 few-shot（完整双场景风格）
- **G6：** G5 + **生活事件时间线约束**；若传入 `life_event_rendering_card` 则注入卡片（见源码分支）
- **G7：** G5 + 由 **`TimelineAgent`** 截断后的生活事件时间线 **纯文本**（与卡片路径不同）
- **G0.5 / G1.5 / G2.0 / G2.5：** 中间消融（如仅事件卡片、无时间线临床、临床+事件卡片等）

另有 **`evaluate_patient_prompts`**、**`generate_patient_prompts_v2`** 及 `G2-1`、`G6-2` 等键名，供 **G-Eval** 复现「模型当时看到的 System Prompt」；以各调用处为准。

**批量 QA 的 JSON 约束（`api.py`）：** 单次输出合法 JSON，如 `{"answers":[{"question_id":1,"answer":"..."}, ...]}`，无 Markdown 围栏；若接口支持可选用 `response_format=json_object`。

### 评测脚本

| 脚本 | 作用 |
|------|------|
| `evaluation/api.py` | **方案 A：** 每个 profile **一次 API 请求**答完 `questions_path` 中全部题目；结果写入 `evaluation/results/{model}_{run_name}_results/run_{baseline}_{run_name}/` |
| `evaluation/run.sh`、`run_qa.sh` | 示例：需先 `export OPENAI_API_KEY`；路径相对本仓库 |
| `evaluation/run_batch.py` | 向本机 HTTP 服务 `POST` 批量任务（需后端已启动） |
| `evaluation/g_eval_new.py` | **LLM裁判**：依赖 `OPENAI_API_KEY`；默认清洗后路径 `evaluation/cleaned_result/{MODEL_NAME}/{baseline}`（可改 `BASELINES_MAP`） |
| `evaluation/post_process.py` | 清洗结果 JSON 中的 `answer`；支持 `--results_parent`、`--cleaned_parent` |

**G-Eval 维度**（见 `g_eval_new.py`）：真实感；人设与大五一致性；事件丰富度与时间多样性；与阳/阴性症状一致性；输出严格 JSON schema。

### 统一患者档案

1. **问诊数据** — 标准化临床症状；主要诊断依据  
2. **咨询数据** — 会话风格与情感反应（仅风格）  
3. **社交媒体** — 纵向事件与症状；构建经历时间线  

两阶段匹配保证人口学与症状跨源一致。

### 症状属性设计

统一标签空间：

- **问诊侧：** 临床 grounded标签的英文表述；文档中正体  
- **社媒侧：** 原始英文标签；文档中 *斜体* 标明来源  

跨模态重叠属性有成对映射。

### Chain-of-Change（CoC）智能体

将噪声时间线转为结构化记忆：

1. 抽取症状/事件节点  
2. 构建带持续关系的时间有序图  
3. 聚合为带相对时间的 **记忆卡片**  

模拟时，提及过去应受检索到的卡片约束，减少无据细节。

### 评测（指标）

- **嵌入类：** 语义真实感、患者间多样性  
- **G-Eval：** 人设忠实度、事件丰富度、症状一致性  
- 消融用于分析各档案组件的贡献  

### 开源清单（建议）

- 发布 **脱敏 profile**、**问题集 JSON**、时间线/卡片脚本或样例、评测与后处理代码。  
- **勿提交** API Key、内网镜像或个人绝对路径；改用环境变量或必填 CLI。  
- 在 README 或 `requirements.txt` 中写明依赖（如 `openai`、`tqdm`、`pandas`）。

### 使用说明

- 仅用于 **研究复现与分析**  
- **不得** 用于真实诊疗  
- 档案均已脱敏，仅供研究  

### 许可

**仅限学术研究使用**（若有单独许可文件以文件为准）。

### 免责声明

本项目用于研究与评测层面的患者行为模拟，**不提供**医疗建议、诊断或治疗建议。
