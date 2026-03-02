#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
implict_test_single.py — single hint-extraction test (LOCAL Qwen3-30B-A3B-Instruct-2507 or DeepSeek API)

This version is adapted for:
  /cluster/home/gw/Backend_project/models/Qwen3-30B-A3B-Instruct-2507

Key upgrades for large local models:
✅ Safe @torch.inference_mode() usage (no invalid @self.torch...)
✅ Supports Qwen chat_template if present (apply_chat_template + add_generation_prompt)
✅ Optional 4-bit / 8-bit loading via bitsandbytes (to fit on GPUs)
✅ Optional attention impl switch (sdpa / flash_attention_2) if your env supports it
✅ Robust device placement even with device_map="auto" (no forcing all tensors to model.device)

Run (LOCAL):
  python3 implict_test_single.py \
    --prompt_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/Implicit_prop/en_hint_extract.txt \
    --corpus_path /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GSC+/corpus/16871364 \
    --local_model_dir /cluster/home/gw/Backend_project/models/Qwen3-30B-A3B-Instruct-2507 \
    --max_chars 1200 \
    --local_load_in_4bit \
    --local_attn_impl sdpa \
    --local_max_new_tokens 512 \
    --local_temperature 0.0

Run (DeepSeek API, optional):
  export DEEPSEEK_API_KEY=...
  python3 implict_test_single.py \
    --prompt_path /path/to/en_hint_extract.txt \
    --corpus_path .../GSC+/corpus/16871364 \
    --base_url https://api.deepseek.com \
    --model deepseek-chat
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import torch


# -------------------------
# Robust protocol parser (UNCHANGED)
# -------------------------
_TAG_RE = re.compile(r"^\s*(SENT_EN|SENTENCE_EN|EN|HINT)\s*[:\t ]\s*(.*)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*END\s*\.?\s*$", re.IGNORECASE)

def _norm_polarity(p: Any) -> str:
    s = str(p or "").strip().lower()
    if s in ("present", "pos", "positive", "yes", "y"):
        return "present"
    if s in ("absent", "neg", "negative", "no", "n", "denied", "without", "not_present"):
        return "absent"
    if s in ("uncertain", "possible", "suspected", "maybe", "probable", "unclear", "may", "might"):
        return "uncertain"
    return "uncertain"

def _split_fields_after_tag(payload: str) -> List[str]:
    # Prefer TAB; fallback to whitespace
    if "\t" in payload:
        parts = [p.strip() for p in payload.split("\t")]
    else:
        parts = [p.strip() for p in re.split(r"\s{2,}|\s+", payload) if p.strip()]
    return parts

def parse_plaintext_protocol(raw: str) -> Tuple[bool, Dict[str, Any], List[str], bool]:
    errs: List[str] = []
    if raw is None:
        return False, {}, ["raw is None"], False

    text = (raw or "").strip("\n")
    if not text.strip():
        return False, {}, ["empty raw"], False

    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, {}, ["empty lines"], False

    sentence_en = ""
    hints: List[Dict[str, str]] = []
    saw_end = False

    for ln in lines:
        if _END_RE.match(ln):
            saw_end = True
            break

        m = _TAG_RE.match(ln)
        if not m:
            continue

        tag = m.group(1).upper()
        payload = (m.group(2) or "").strip()

        if tag in ("SENT_EN", "SENTENCE_EN", "EN"):
            if payload:
                sentence_en = payload
            else:
                errs.append("SENT_EN missing value")
            continue

        if tag == "HINT":
            parts = _split_fields_after_tag(payload)
            if len(parts) < 3:
                errs.append(f"HINT bad fields: {ln}")
                continue
            polarity = _norm_polarity(parts[-2])
            tp = (parts[-1] or "other").strip().lower() or "other"
            hint_en = " ".join(parts[:-2]).strip()
            if not hint_en:
                errs.append(f"HINT empty hint_en: {ln}")
                continue
            hints.append({"hint_en": hint_en, "polarity": polarity, "type": tp})

    if not sentence_en.strip():
        errs.append("missing sentence_en")
    if not saw_end:
        errs.append("missing END sentinel")

    ok = (sentence_en.strip() != "")
    return ok, {"sentence_en": sentence_en.strip(), "hints": hints}, errs, saw_end


# -------------------------
# DeepSeek API hint client (kept)
# -------------------------
class DeepSeekHintClient:
    def __init__(self, api_key: str, base_url: str, model: str, prompt_path: str, timeout: float = 60.0):
        import requests  # local import to avoid dependency if unused
        self._requests = requests

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)

        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"prompt_path not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def render_prompt(self, text: str) -> str:
        p = self.prompt_template
        p = p.replace("{text}", text or "")
        p = p.replace("{{text}}", text or "")
        p = p.replace("{{context}}", text or "")
        return p

    def call(self, text: str) -> str:
        prompt = self.render_prompt(text)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        resp = self._requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# -------------------------
# LOCAL HF hint client (Qwen3-30B-A3B ready)
# -------------------------
class LocalHFHintClient:
    """
    Uses Transformers AutoTokenizer + AutoModelForCausalLM.

    For Qwen Instruct models, chat_template is often available:
      tokenizer.apply_chat_template([...], add_generation_prompt=True)

    Large-model options:
      - load_in_4bit / load_in_8bit (bitsandbytes)
      - attn_impl: "sdpa" or "flash_attention_2" (if installed) or None
      - dtype: bf16/fp16/fp32
    """
    def __init__(
        self,
        model_dir: str,
        prompt_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        dtype: str = "bf16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_impl: str = "sdpa",
    ):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"local_model_dir not found: {model_dir}")
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_dir = model_dir
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        dtype = (dtype or "bf16").lower().strip()
        if dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype in ("fp16", "float16", "half"):
            torch_dtype = torch.float16
        elif dtype in ("fp32", "float32"):
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown --local_dtype: {dtype} (use bf16/fp16/fp32)")

        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose only one: --local_load_in_4bit OR --local_load_in_8bit")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            trust_remote_code=True,
        )

        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Optional attention implementation (if your env supports it)
        attn_impl = (attn_impl or "").strip().lower()
        if attn_impl and attn_impl != "none":
            # transformers supports attn_implementation in many causal LM classes
            model_kwargs["attn_implementation"] = attn_impl

        # Optional quantization (requires bitsandbytes)
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except Exception as e:
                raise RuntimeError(
                    "bitsandbytes quantization requested but transformers BitsAndBytesConfig unavailable. "
                    "Install bitsandbytes + a compatible transformers build."
                ) from e

            if load_in_4bit:
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            else:
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

            model_kwargs["quantization_config"] = bnb_cfg

        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
        self.model.eval()

        # Some tokenizers don't have pad_token set
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def render_prompt(self, text: str) -> str:
        p = self.prompt_template
        p = p.replace("{text}", text or "")
        p = p.replace("{{text}}", text or "")
        p = p.replace("{{context}}", text or "")
        return p

    def _has_chat_template(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None)) and hasattr(self.tokenizer, "apply_chat_template")

    def _build_inputs(self, prompt: str) -> Dict[str, Any]:
        # IMPORTANT for device_map="auto": DO NOT force tensors to a single device.
        # Passing CPU tensors is fine; HF will dispatch during generate.
        if self._has_chat_template():
            messages = [{"role": "user", "content": prompt}]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = self.tokenizer(rendered, return_tensors="pt", padding=False, truncation=True)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)
        return enc

    @torch.inference_mode()
    def call(self, text: str) -> str:
        prompt = self.render_prompt(text)
        inputs = self._build_inputs(prompt)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0.0),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0][prompt_len:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return (decoded or "").strip()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_path", type=str, required=True, help="en_hint_extract.txt")
    ap.add_argument("--corpus_path", type=str, required=True, help=".../GSC+/corpus/16871364")
    ap.add_argument("--max_chars", type=int, default=1200, help="truncate corpus text for test call")

    # API mode (optional)
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-chat")

    # Local mode
    ap.add_argument("--local_model_dir", type=str, default="", help="e.g. /cluster/home/gw/Backend_project/models/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--local_max_new_tokens", type=int, default=512)
    ap.add_argument("--local_temperature", type=float, default=0.0)
    ap.add_argument("--local_top_p", type=float, default=1.0)

    # Large-model controls
    ap.add_argument("--local_dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    ap.add_argument("--local_load_in_4bit", action="store_true", help="Enable 4-bit quantization (bitsandbytes)")
    ap.add_argument("--local_load_in_8bit", action="store_true", help="Enable 8-bit quantization (bitsandbytes)")
    ap.add_argument("--local_attn_impl", type=str, default="sdpa", help="sdpa|flash_attention_2|none")

    args = ap.parse_args()

    # load corpus text
    if not os.path.isfile(args.corpus_path):
        raise FileNotFoundError(f"corpus_path not found: {args.corpus_path}")
    with open(args.corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    text_in = (corpus_text or "").strip()
    if args.max_chars > 0 and len(text_in) > args.max_chars:
        text_in = text_in[: args.max_chars]

    # Choose client: LOCAL if provided, else API
    if args.local_model_dir.strip():
        client = LocalHFHintClient(
            model_dir=args.local_model_dir.strip(),
            prompt_path=args.prompt_path,
            max_new_tokens=args.local_max_new_tokens,
            temperature=args.local_temperature,
            top_p=args.local_top_p,
            dtype=args.local_dtype,
            load_in_4bit=args.local_load_in_4bit,
            load_in_8bit=args.local_load_in_8bit,
            attn_impl=args.local_attn_impl,
        )
        client_name = f"LOCAL({args.local_model_dir.strip()})"
    else:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Env var {args.api_key_env} not set AND no --local_model_dir provided.\n"
                f"Either:\n  export {args.api_key_env}=...\nOr:\n  --local_model_dir /cluster/home/gw/Backend_project/models/Qwen3-30B-A3B-Instruct-2507"
            )
        client = DeepSeekHintClient(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            prompt_path=args.prompt_path,
            timeout=60.0,
        )
        client_name = f"API({args.base_url}, {args.model})"

    prompt_preview = client.render_prompt(text_in)
    print(f"=== CLIENT: {client_name} ===")
    print("=== PROMPT PREVIEW (first 800 chars) ===")
    print(prompt_preview[:800])
    print("\n=== CALLING LLM ===")

    raw = client.call(text_in)

    print("\n=== RAW LLM OUTPUT ===")
    print(raw)

    ok, parsed, errs, saw_end = parse_plaintext_protocol(raw)

    # If END missing, try append once (common)
    if not saw_end:
        raw2 = (raw or "").rstrip() + "\nEND\n"
        ok2, parsed2, errs2, saw_end2 = parse_plaintext_protocol(raw2)
        if ok2 and saw_end2:
            ok, parsed, errs, saw_end = ok2, parsed2, errs2, saw_end2

    print("\n=== PARSE RESULT ===")
    print("ok:", ok, "saw_end:", saw_end)
    if errs:
        print("errs:", errs[:10])

    print("\nSENT_EN:")
    print(parsed.get("sentence_en", ""))

    print("\nHINTS:")
    for h in parsed.get("hints", [])[:30]:
        print("-", json.dumps(h, ensure_ascii=False))


if __name__ == "__main__":
    main()
