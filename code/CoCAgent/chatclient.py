import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from dataclasses import dataclass, field


def _default_hf_model() -> str:
    return os.getenv("DEPROFILE_MODEL_DIR", "Qwen/Qwen3-4B-Instruct-2507")


@dataclass
class HFQwenChatClient:
    model_name_or_path: str = field(default_factory=_default_hf_model)
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    max_context: Optional[int] = None
    cuda_index: int = 0

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            device_map={"": self.cuda_index},     # 或 device_map="cuda:0"
        )

        self.model.eval()
        self.has_chat_template = hasattr(self.tokenizer, "apply_chat_template")

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        if self.has_chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        # fallback
        chunks = []
        for m in messages:
            chunks.append(f"[{m['role'].upper()}]\n{m['content'].strip()}\n")
        chunks.append("[ASSISTANT]\n")
        return "\n".join(chunks)

    @torch.inference_mode()
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if self.max_context is not None and inputs["input_ids"].shape[1] > self.max_context:
            inputs["input_ids"] = inputs["input_ids"][:, -self.max_context:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_context:]

        do_sample = temperature > 0.0
        gen = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = gen[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
