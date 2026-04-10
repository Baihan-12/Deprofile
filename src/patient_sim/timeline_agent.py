# Timeline loading and string rendering for longitudinal STMHD-style JSON.
import json
import os


def _default_data_root() -> str:
    return os.environ.get(
        "DEPROFILE_DATA_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
    )


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


class TimelineAgent:
    def __init__(self, profile, candidate_index, timeline_type="life_event", timestamp=None):
        self.profile = profile
        self.candidate_index = candidate_index
        self.timeline_type = timeline_type
        self.timeline_dir = os.path.join(
            _default_data_root(), f"stmhd_{self.timeline_type}_timeline"
        )
        self.tweet_user_id = None
        self.timeline = None

    def _get_tweet_user_id(self):
        if not self.profile["candidate_id"]:
            return
        if self.candidate_index is None or self.candidate_index >= len(self.profile["candidate_id"]):
            self.candidate_index = 0
        self.tweet_user_id = self.profile["candidate_id"][self.candidate_index]["basic_id"]

    def get_naive_timeline(self):
        self._get_tweet_user_id()
        if not self.tweet_user_id:
            return
        with open(os.path.join(self.timeline_dir, f"{self.tweet_user_id}.json"), "r") as f:
            self.timeline = json.load(f)["timeline"]
        return self.timeline

    def get_cut_timeline(self, timestamp=None, max_events_num=50):
        self.get_naive_timeline()
        if not self.timeline:
            return
        if timestamp is None:
            timestamp = self.timeline[-1]["timestamp"]
        if max_events_num is None:
            max_events_num = len(self.timeline)
        cut_timeline = []
        current_index = -1
        while (
            -current_index < len(self.timeline)
            and current_index < max_events_num
            and self.timeline[current_index]["timestamp"] > timestamp - 90
        ):
            cut_timeline.append(self.timeline[current_index])
            current_index -= 1
        return " ".join(
            [
                f"{timestamp - event['timestamp']} days ago: {event[self.timeline_type]}-{event['tweet']}"
                for event in cut_timeline
            ]
        )

    def get_timeline_summary(self, timeline):
        return " ".join([event["content"] for event in timeline])
