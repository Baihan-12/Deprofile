# STMHD 时间线数据

本目录包含论文中使用的 **STMHD** 标注时间线，与 `ACL_agent/data` 下结构一致：

| 子目录 | 内容 |
|--------|------|
| `stmhd_symptom_timeline/` | 症状相关时间线；每条用户一个 JSON；根目录含 `user_symptom_statistics.csv` |
| `stmhd_life_event_timeline/` | 生活事件时间线；含 `user_statistics.csv` 等汇总 |

评测代码中的 `TimelineAgent`（见 `codes/evaluation/agent.py`）默认读取本目录下 `stmhd_{symptom|life_event}_timeline`。
