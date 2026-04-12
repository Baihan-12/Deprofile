# 仓库规划

请整理 `/aistor/sjtu/hpc_stor01/home/libaihan/deprofile` 文件夹下的内容，不需要修改原来的文件，只需要把符合要求或者修改后的文件放到 `Deprofile` 文件夹中，本仓库是针对 `Deprofile_emnlp.pdf` 的 GitHub 仓库。

本仓库会包含以下五个方面的内容：

## 1. DEPROFILE 文件夹

- `deprofiles_complete_index.json`：deprofile 完整的所有 pair。
- **已完成：** 统计 deprofile 各方面数据并写入 `DEPROFILE/dataset_statistics.json` 与根目录 `Readme.md`。
- **已完成：** `selected_samples.json` 仅保留 pair 编号：69, 91, 99, 107, 120, 151, 559, 563, 767, 770, 911, 1008, 1100, 1136, 1506, 1681, 1795, 1961, 2062, 2310, 2556, 2599, 2652, 2737, 2798, 2805, 2960（共 27 条）。
- **已完成：** 从 `ACL_agent/evaluation/dialogues_gt` 筛选上述 pair 的 assessment（问诊）与 counseling（咨询）对话至 `DEPROFILE/dialogues_sample/`。

## 2. timeline

- **已完成：** STMHD 标注时间线置于 `timeline/stmhd_symptom_timeline/` 与 `timeline/stmhd_life_event_timeline/`（含 CSV 统计），说明见 `timeline/README.md`。

## 3. prompts

- **labeling：** `prompts/labeling/`（来自 `deprofile/codes/` 的标注与流水线脚本）。
- **patient：** `prompts/patient/`（来自 `ACL_agent/evaluation/prompts/` 的最终 deprofile prompt 片段）。
- **evaluation：** `prompts/evaluation/`（G-Eval 模板摘要 + `questions/` 下全部 QA JSON）。

## 4. codes

- **已完成：** `codes/pair/`（配对与预处理）、`codes/evaluation/`（评测与 `utils.generate_patient_prompts`）、`codes/CoCAgent/`（CoC 智能体）。

---

**维护：** 若需从上级目录重新同步布局，在仓库根执行：`python3 scripts/build_repo_layout.py`（再按需手动调整 CoC `__main__` 中的模型路径等）。
