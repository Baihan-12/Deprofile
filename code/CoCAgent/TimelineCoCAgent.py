import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

_DEFAULT_REPO_DATA = Path(__file__).resolve().parents[2] / "data"
_DEFAULT_TIMELINE_ROOT = str(_DEFAULT_REPO_DATA)
_DEFAULT_TIMELINE_MEMORY_PREFIX = str(_DEFAULT_REPO_DATA / "timeline_memory") + os.sep
from collections import defaultdict
from utils import days_to_relative_cn, clamp_text, safe_json_loads
from chatclient import HFQwenChatClient

EXTRACT_EVENT_PROMPT_ZH = """/no_thinking你是一名社交媒体事件信息抽取专家。下面是一条与【{item}】相关的推文：

{tweet}

任务：判断这条推文是否包含“对用户本人有意义、且由用户亲身经历/直接相关”的生活事件。
- 如果不包含（例如：只是评价别人、辱骂/标签化他人、转述他人经历、无明确事件、纯情绪宣泄且无事件），请仅输出：None

如果包含，请抽取一个“事件三元组”，并给出一句中文摘要。
严格输出一个 JSON（不要输出任何额外文字），格式如下：
{{
  "is_meaningful": true,
  "event_triple": "<主体> <谓语> <宾语>",
  "event_summary_cn": ""
}}

要求：
1) 三元组要尽量贴近原文，不要编造细节（人物/地点/时间/原因）。
2) 主体优先用“我/用户本人”，除非原文明确是他人且与用户无关则判 None。
3) event_summary_cn 用一句话概括事件（不超过25个汉字）。
"""

EXTRACT_SYMPTOM_PROMPT_ZH = """/no_thinking你是一名社交媒体【症状/体验信息】抽取专家。下面是一条与【{item}】相关的推文：

{tweet}

任务：判断这条推文是否包含“对用户本人有意义、且由用户亲身体验/直接相关”的症状、主观不适或功能受损（symptom/complaint/impairment）。

- 如果不包含（例如：只是评价别人、辱骂/标签化他人、转述他人经历、无明确身心体验、纯情绪宣泄但没有任何身心/行为/功能表现），请仅输出：None

如果包含，请抽取一个“症状三元组”，并给出一句中文摘要。
严格输出一个 JSON（不要输出任何额外文字），格式如下：
{{
  "is_meaningful": true,
  "event_triple": "<主体> <谓语> <宾语>",
  "event_summary_cn": ""
}}

要求：
1) 三元组要尽量贴近原文，不要编造细节（具体部位/频率/原因/持续时间等），除非原文明确提到。
2) 主体优先用“我/用户本人”。如果原文明显是他人的体验且与用户无关，则输出 None。
3) 谓语优先用“出现/感到/受到影响/难以进行”。选择最贴近原文的一个。
4) 宾语必须是“更具体的症状/体验短语”，应基于 tweet 原文改写为中文短语（例如“胃不舒服/反胃/腹痛”“精力明显下降”“入睡困难”“对社交提不起兴趣”“做事很难集中注意力”），不要只输出 {item} 这个标签名。
   - 但可以把 {item} 作为语义约束：如果 tweet 内容与 {item} 完全不一致，则输出 None。
5) symptom_summary_cn 用一句话概括症状体验（不超过25个汉字），不要写诊断名，不要科普。
"""



EXTRACT_EVENT_PROMPT = """You are a social media event information extraction expert. Here is a tweet about {item}:
{tweet}

First, determine whether the event related to the {item} mentioned in this tweet is meaningful to the user.
If not meaningful, output exactly: None

If meaningful, extract the main RDF-style event triple and provide structured metadata fields.
Strictly output ONE valid JSON object with the following format, and NOTHING else:
{{
  "event_triple": "<subject> <predicate> <object>",
  "event_type": "",
  "emotion": "",
  "time_expression": "",
  "location_expression": "",
  "external_events": "",
  "related_context": "",
  "surface_variants": [""],
  "user_role": ""
}}
"""


# =========================================================
# 2) CoC Timeline Agent (Graph + Window Episodes + Rule Cards)
# =========================================================
class TimelineAgentCoC:
    """
    - timestamp is day index (int)
    - now_day defaults to last timestamp in the file (or user provided)
    - life_event nodes: call LLM extract_event
    - symptom nodes: no LLM
    - episodes: fixed window bucketing (e.g., 7 or 14 days)
    - cards: rule-based rendering from episodes + nodes (NO LLM)
    """

    def __init__(
        self,
        profile: Dict[str, Any],
        candidate_index: int,
        timeline_type: str = "life_event",  # "life_event" or "symptom"
        timeline_root: str = _DEFAULT_TIMELINE_ROOT,
        qwen_client: Optional[HFQwenChatClient] = None,
    ):
        assert timeline_type in ["life_event", "symptom"]
        self.profile = profile
        self.candidate_index = candidate_index
        self.timeline_type = timeline_type
        self.timeline_dir = os.path.join(timeline_root, f"stmhd_{timeline_type}_timeline")
        self.qwen = qwen_client

        self.tweet_user_id: Optional[str] = None
        self.timeline: Optional[List[Dict[str, Any]]] = None

    def _get_tweet_user_id(self) -> None:
        cand = self.profile.get("candidate_id")
        if not cand:
            return
        if self.candidate_index is None or self.candidate_index >= len(cand):
            self.candidate_index = 0
        self.tweet_user_id = cand[self.candidate_index]["basic_id"]

    def load_timeline(self) -> List[Dict[str, Any]]:
        self._get_tweet_user_id()
        if not self.tweet_user_id:
            return []
        path = os.path.join(self.timeline_dir, f"{self.tweet_user_id}.json")
        with open(path, "r") as f:
            self.timeline = json.load(f)["timeline"]
        self.timeline.sort(key=lambda x: x["timestamp"])
        return self.timeline

    def cut_items(
        self,
        now_day: Optional[int] = None,
        horizon_days: int = 90,
        max_events_num: int = 80,
    ) -> Tuple[List[Dict[str, Any]], int]:
        tl = self.load_timeline()
        if not tl:
            return [], 0
        if now_day is None:
            now_day = tl[-1]["timestamp"]

        items = [x for x in tl if (x["timestamp"] <= now_day and x["timestamp"] > now_day - horizon_days)]
        if max_events_num is not None and len(items) > max_events_num:
            items = items[-max_events_num:]
        return items, now_day

    # -----------------------------
    # Graph builder
    # -----------------------------
    def build_graph(
        self,
        now_day: Optional[int] = None,
        horizon_days: int = 90,
        max_events_num: int = 80,
        use_llm_for_life_event: bool = True,
        use_llm_for_symptom: bool = True,
    ) -> Dict[str, Any]:
        items, now_day = self.cut_items(now_day, horizon_days, max_events_num)

        graph = {
            "time_axis": {"anchor_day": now_day, "unit": "day", "timezone": "Europe/Paris"},
            "timeline_type": self.timeline_type,
            "nodes": [],
            "edges": [],
            "index": {"by_day": defaultdict(list), "by_symptom": defaultdict(list)}
        }

        it = tqdm(items, desc=f"CoC graph nodes ({self.timeline_type})", leave=False)
        for ev in it:
            day = int(ev["timestamp"])
            days_ago = int(now_day - day)

            time_norm = {
                "event_day": day,
                "days_ago": days_ago,
                "relative_cn": days_to_relative_cn(days_ago),
                "absolute_date": None,
                "granularity": "day",
                "confidence": 1.0
            }

            tweet = ev.get("tweet", "")

            if self.timeline_type == "symptom" :
                if use_llm_for_symptom and self.qwen is None:
                    raise RuntimeError("use_llm_for_symptom=True but qwen_client is None.")
                
                symptom = ev.get("symptom")

                if use_llm_for_symptom:
                    prompt = EXTRACT_SYMPTOM_PROMPT_ZH.format(item=symptom, tweet=tweet)
                    out = self.qwen.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
                    # print(out)
                    if out.strip() == "None":
                        extracted = None
                    else:
                        # print(out)
                        extracted = safe_json_loads(out)
                        if not extracted.get("is_meaningful", False):
                            extracted = None
                
                    if extracted is None:
                        continue
                    
                        
                node_id = f"SYM_{day}_{symptom}"
                node = {
                    "id": node_id,
                    "node_type": "Symptom",
                    "timestamp_day": day,
                    "symptom": symptom,
                    "triple": extracted.get("event_triple"),
                    "evidence": [tweet],
                    "time_norm": time_norm
                }
                graph["nodes"].append(node)
                graph["index"]["by_day"][day].append(node_id)
                graph["index"]["by_symptom"][symptom].append(node_id)

            else:
                # life_event: LLM extract_event
                if use_llm_for_life_event and self.qwen is None:
                    raise RuntimeError("use_llm_for_life_event=True but qwen_client is None.")

                life_event_label = ev.get("life_event", ev.get("content", "Unknown"))
                node_id = f"EV_{day}_{len(graph['nodes'])}"

                extracted = None
                if use_llm_for_life_event:
                    prompt = EXTRACT_EVENT_PROMPT_ZH.format(item=life_event_label, tweet=tweet)
                    out = self.qwen.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512)
                    if out.strip() == "None":
                        extracted = None
                    else:
                        extracted = safe_json_loads(out)
                        if not extracted.get("is_meaningful", False):
                            extracted = None

                    # 如果无效：直接 continue（不加入 nodes）
                    if extracted is None:
                        continue
                    if not extracted.get("is_meaningful", False):
                        continue
                    if not extracted.get("event_triple"):
                        continue


                node = {
                        "id": node_id,
                        "node_type": "LifeEvent",
                        "timestamp_day": day,
                        "time_norm": time_norm,
                        "is_meaningful": True,
                        "event_triple": extracted.get("event_triple"),
                        "event_summary_cn": extracted.get("event_summary_cn")
                    }

                graph["nodes"].append(node)
                graph["index"]["by_day"][day].append(node_id)

        # edges: basic temporal skeleton + symptom persists
        nodes_sorted = sorted(graph["nodes"], key=lambda x: x["timestamp_day"])
        for a, b in zip(nodes_sorted, nodes_sorted[1:]):
            graph["edges"].append({
                "source": a["id"],
                "target": b["id"],
                "relation": "temporal_precedes",
                "confidence": 0.5,
                "rationale": "按时间顺序"
            })

        if self.timeline_type == "symptom":
            for sym, ids in graph["index"]["by_symptom"].items():
                if len(ids) >= 2:
                    for u, v in zip(ids, ids[1:]):
                        graph["edges"].append({
                            "source": u,
                            "target": v,
                            "relation": "persists",
                            "confidence": 0.75,
                            "rationale": "同症状再次出现"
                        })

        # jsonify defaultdict
        graph["index"]["by_day"] = {str(k): v for k, v in graph["index"]["by_day"].items()}
        graph["index"]["by_symptom"] = dict(graph["index"]["by_symptom"]) if self.timeline_type == "symptom" else None
        return graph

    # -----------------------------
    # Episodes by window (no LLM)
    # -----------------------------
    @staticmethod
    def build_episodes_by_window(graph: Dict[str, Any], window_days: int = 7) -> List[Dict[str, Any]]:
        anchor_day = int(graph["time_axis"]["anchor_day"])
        nodes = graph["nodes"]

        buckets = defaultdict(list)  # bucket_id -> node_ids
        node_map = {n["id"]: n for n in nodes}

        for n in nodes:
            days_ago = int(n["time_norm"]["days_ago"])
            bucket_id = days_ago // window_days  # 0 = latest window
            buckets[bucket_id].append(n["id"])

        episodes = []
        for bid in sorted(buckets.keys()):
            start_ago = bid * window_days + (window_days - 1)
            end_ago = bid * window_days
            episodes.append({
                "episode_id": f"E_{bid}",
                "window_days": window_days,
                "time_range": {
                    "days_ago_start": int(start_ago),
                    "days_ago_end": int(end_ago),
                    "relative_cn": f"{days_to_relative_cn(start_ago)}~{days_to_relative_cn(end_ago)}"
                },
                "salient_node_ids": buckets[bid],
            })

        # 你也可以按“节点数量/重要性”裁剪每个 episode 的 salient_node_ids
        return episodes

    # -----------------------------
    # Cards rendering (rule-based, NO LLM)
    # -----------------------------
    @staticmethod
    def render_cards(graph: Dict[str, Any], episodes: List[Dict[str, Any]], max_symptoms: int = 3, max_events: int = 2) -> List[Dict[str, Any]]:
        node_map = {n["id"]: n for n in graph["nodes"]}

        cards = []
        for ep in episodes:
            node_ids = ep["salient_node_ids"]
            symptoms = []
            events = []

            # collect
            for nid in node_ids:
                n = node_map[nid]
                if n["node_type"] == "Symptom":
                    symptoms.append(n)
                elif n["node_type"] == "LifeEvent":
                    events.append(n)

            # pick top symptoms/events (simple: most recent within episode)
            symptoms.sort(key=lambda x: x["timestamp_day"], reverse=True)
            events.sort(key=lambda x: x["timestamp_day"], reverse=True)

            top_sym = [s.get("symptom") for s in symptoms[:max_symptoms] if s.get("symptom")]
            top_evt = []
            for e in events[:max_events]:
                # prefer event_conclusion-like label if exists, else short evidence
                label = e.get("life_event") or e.get("event_type") or clamp_text(e["evidence"][0], 40)
                top_evt.append(label)

            # compose: must include time range + (optional) one concrete timepoint
            # choose a representative node for concrete timepoint (most recent node in episode)
            rep_node = None
            if symptoms:
                rep_node = symptoms[0]
            elif events:
                rep_node = events[0]

            if rep_node is not None:
                rep_days = rep_node["time_norm"]["days_ago"]
                rep_rel = rep_node["time_norm"]["relative_cn"]
                rep_time = f"{rep_rel}（{rep_days}天前）"
            else:
                rep_time = ep["time_range"]["relative_cn"]

            parts = []
            parts.append(f"{ep['time_range']['relative_cn']}：")

            if top_sym:
                parts.append("主要是" + "、".join(top_sym))
            if top_evt:
                parts.append("相关事件：" + "；".join(top_evt))

            # 如果都空，保底写 evidence
            if len(parts) == 1 and node_ids:
                ev = node_map[node_ids[-1]].get("evidence", [""])[0]
                parts.append(clamp_text(ev, 60))

            card_cn = "；".join(parts)
            # 强制塞一个可追问的具体时间点（代表点）
            card_cn = f"{card_cn}。代表时间点：{rep_time}。"

            cards.append({
                "episode_id": ep["episode_id"],
                "time_range": ep["time_range"],
                "salient_node_ids": node_ids,
                "card_cn": card_cn
            })

        return cards

    @staticmethod
    def render_cards_minimal(
    graph: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    max_symptoms: int = 3,
    max_events: int = 2
) -> List[Dict[str, Any]]:
        """
        Minimal card rendering:
        - Same inputs as before (graph with nodes+id; episodes with salient_node_ids)
        - Output does NOT include node ids
        - No evidence/emotion
        - Prefer event_triple / event_summary_cn for life events
        """
        node_map = {n["id"]: n for n in graph.get("nodes", [])}

        cards = []
        for ep in episodes:
            node_ids = ep.get("salient_node_ids", [])
            if not node_ids:
                # 仍然输出一条空卡片，避免下游崩
                cards.append({
                    "episode_id": ep.get("episode_id"),
                    "time_range": ep.get("time_range"),
                    "card_cn": f"{ep.get('time_range', {}).get('relative_cn', '')}：无有效信息。"
                })
                continue

            symptoms = []
            events = []

            # collect (still uses ids internally)
            for nid in node_ids:
                n = node_map.get(nid)
                if not n:
                    continue
                if n.get("node_type") == "Symptom":
                    symptoms.append(n)
                elif n.get("node_type") == "LifeEvent":
                    events.append(n)

            # sort by recency
            symptoms.sort(key=lambda x: x.get("timestamp_day", -1), reverse=True)
            events.sort(key=lambda x: x.get("timestamp_day", -1), reverse=True)

            # representative timepoint: pick the most recent node among all
            rep_node = None
            all_nodes = symptoms + events
            if all_nodes:
                rep_node = max(all_nodes, key=lambda x: x.get("timestamp_day", -1))

            if rep_node is not None and rep_node.get("time_norm"):
                rep_days = rep_node["time_norm"].get("days_ago")
                rep_rel = rep_node["time_norm"].get("relative_cn")
                rep_time = f"{rep_rel}（{rep_days}天前）"
            else:
                rep_time = ep.get("time_range", {}).get("relative_cn", "")

            # build minimal lines (triples / one-liners)
            lines = []
            seen = set()
            # symptom lines
            for s in symptoms[:max_symptoms]:

                tri = s.get("triple")
                summ = s.get("summary_cn")
                label = s.get("symptom")

                text = tri or summ or (f"<我> <出现> <{label}>" if label else None)
                if not text:
                    continue
                if text in seen:
                    continue
                seen.add(text)

                lines.append(clamp_text(text, 60))

            # event lines (prefer triple/summary; no evidence)
            for e in events[:max_events]:
                tri = e.get("event_triple")
                summ = e.get("event_summary_cn")
                fallback = e.get("life_event") or e.get("event_type")
                text = tri or summ or fallback
                if not text:
                    continue
                # 保持一句话，不要太长
                lines.append(clamp_text(text, 60))

            # compose card text
            prefix = ep.get("time_range", {}).get("relative_cn", "")
            if lines:
                card_cn = f"{prefix}：{'；'.join(lines)}。代表时间点：{rep_time}。"
            else:
                card_cn = f"{prefix}：无有效信息。代表时间点：{rep_time}。"

            # IMPORTANT: no salient_node_ids in output
            cards.append({
                "episode_id": ep.get("episode_id"),
                "time_range": ep.get("time_range"),
                "card_cn": card_cn
            })

        return cards


    def write_to_profile_memory(self, memory_key: str, payload: Dict[str, Any]) -> None:
        self.profile.setdefault("memory", {})
        self.profile["memory"][memory_key] = payload

    def save_graph_and_episodes_and_cards(self, profile_id: str, save_dir: str, payload: Dict[str, Any]) -> None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{profile_id}.json"), "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)

# =========================================================
# 5) One-call builder: write (graph + episodes + cards) into profile memory
# =========================================================
def build_profile_timeline_memory_coc(
    profile: Dict[str, Any],
    candidate_index: int = 0,
    now_day: Optional[int] = None,
    horizon_days: int = 90,
    max_events_num: int = 80,
    window_days: int = 7,
    timeline_root: str = _DEFAULT_TIMELINE_ROOT,
    qwen: Optional[HFQwenChatClient] = None,
    use_llm_for_life_event: bool = True,
    use_llm_for_symptom: bool = True,
    profile_id: str = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Output written to:
      profile["memory"]["symptom_timeline_memory"] = {graph, episodes, cards}
      profile["memory"]["life_event_timeline_memory"] = {graph, episodes, cards}
    """
    save_dir = _DEFAULT_TIMELINE_MEMORY_PREFIX
    # 如果最后的json存在直接continue
    if os.path.exists(os.path.join(save_dir+"life_event/", f"{profile_id}_{profile['candidate_id'][candidate_index]['basic_id']}.json")):
        return None
    if os.path.exists(os.path.join(save_dir+"symptom/", f"{profile_id}_{profile['candidate_id'][candidate_index]['basic_id']}.json")):
        return None
    # -------- symptom --------
    sym_agent = TimelineAgentCoC(
        profile=profile,
        candidate_index=candidate_index,
        timeline_type="symptom",
        timeline_root=timeline_root,
        qwen_client=qwen
    )
    sym_graph = sym_agent.build_graph(
        now_day=now_day,
        horizon_days=horizon_days,
        max_events_num=max_events_num,
        use_llm_for_life_event=False
    )
    # if debug:
    #     print("sym_graph:","="*100)
    #     print(sym_graph)
    sym_episodes = TimelineAgentCoC.build_episodes_by_window(sym_graph, window_days=window_days)
    # if debug:
    #     print("sym_episodes:","="*100)
    #     print(sym_episodes)
    sym_cards = TimelineAgentCoC.render_cards_minimal(sym_graph, sym_episodes)
    sym_payload = {"graph": sym_graph, "episodes": sym_episodes, "cards": sym_cards}
    if debug:
        print("sym_cards:","="*100)
        print(sym_cards)
    # if not debug:
    #     sym_agent.write_to_profile_memory("symptom_timeline_memory", sym_payload)

    # -------- life_event --------
    ev_agent = TimelineAgentCoC(
        profile=profile,
        candidate_index=candidate_index,
        timeline_type="life_event",
        timeline_root=timeline_root,
        qwen_client=qwen
    )
    ev_graph = ev_agent.build_graph(
        now_day=now_day,
        horizon_days=horizon_days,
        max_events_num=max_events_num,
        use_llm_for_life_event=use_llm_for_life_event
    )
    if debug:
        print("ev_graph:","="*100)
        print(ev_graph)
    ev_episodes = TimelineAgentCoC.build_episodes_by_window(ev_graph, window_days=window_days)
    if debug:
        print("ev_episodes:","="*100)
        print(ev_episodes)
    ev_cards = TimelineAgentCoC.render_cards_minimal(ev_graph, ev_episodes)
    if debug:
        print("ev_cards:","="*100)
        print(ev_cards)
    ev_payload = {"graph": ev_graph, "episodes": ev_episodes, "cards": ev_cards}
 
    # if not debug:
    #     ev_agent.write_to_profile_memory("life_event_timeline_memory", ev_payload)

    save_dir = _DEFAULT_TIMELINE_MEMORY_PREFIX
    ev_agent.save_graph_and_episodes_and_cards(profile_id=f"{profile_id}_{profile['candidate_id'][candidate_index]['basic_id']}", save_dir=save_dir+"life_event/", payload=ev_payload)
    sym_agent.save_graph_and_episodes_and_cards(profile_id=f"{profile_id}_{profile['candidate_id'][candidate_index]['basic_id']}", save_dir=save_dir+"symptom/", payload=sym_payload)

    return None

if __name__ == "__main__":
    cuda_index = 2
    qwen = HFQwenChatClient(
        model_name_or_path=os.getenv("DEPROFILE_MODEL_DIR", "Qwen/Qwen3-8B"),
        cuda_index=cuda_index,
    )
    debug = False
    profilejson = json.load(
        open(_DEFAULT_REPO_DATA / "main_select_profiles_2.json", "r", encoding="utf-8")
    )

    # start_index = 1*len(profilejson.keys())//4
    # end_index = 2*len(profilejson.keys())//4+1
    for idx in tqdm(list(profilejson.keys())):
        profile = profilejson[idx]
        mem = build_profile_timeline_memory_coc(
            profile=profile,
            candidate_index=0,
            now_day=None,           # 默认 timeline 最后一天；你也可以传一个过去的 day 做回放
            horizon_days=90,
            max_events_num=80,
            window_days=7,
            debug=debug,
            qwen=qwen,
            profile_id=idx,
        )
