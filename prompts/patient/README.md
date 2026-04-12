# 患者模拟 Prompt 片段

- **`clinical.py`**：按症状标签组织的阳性/阴性临床描述模板；`SYMPTOM_TIMELINE_DIR` 已指向仓库内 `timeline/stmhd_symptom_timeline`。
- **`risk.py`**：风险相关文案。
- **`personality_traits.py`**：大五人格相关 prompt 片段。

评测主逻辑在 `codes/evaluation/utils.py` 中组装完整 system prompt（G0–G7 等）。
