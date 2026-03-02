#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
implict_test_single.py — single hint-extraction test (standalone client)

- Builds a dedicated DeepSeek client ONLY for hint extraction (no refiner client)
- Uses en_hint_extract.txt
- Tests one GSC+ corpus file: .../GSC+/corpus/16871364
- Prints:
  - rendered prompt preview
  - raw LLM output
  - parsed SENT_EN + HINT lines
"""

import os
import re
import json
import argparse
import requests
from typing import Any, Dict, List, Tuple

# -------------------------
# Robust protocol parser
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
# Dedicated hint LLM client
# -------------------------
class DeepSeekHintClient:
    def __init__(self, api_key: str, base_url: str, model: str, prompt_path: str, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)

        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"prompt_path not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def render_prompt(self, text: str) -> str:
        # Support {text} and {{text}} and also {{context}} as alias
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
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-chat")
    ap.add_argument("--prompt_path", type=str, required=True, help="en_hint_extract.txt")
    ap.add_argument("--corpus_path", type=str, required=True, help=".../GSC+/corpus/16871364")
    ap.add_argument("--max_chars", type=int, default=1200, help="truncate corpus text for test call")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Env var {args.api_key_env} not set. export {args.api_key_env}=...")

    # load corpus text
    if not os.path.isfile(args.corpus_path):
        raise FileNotFoundError(f"corpus_path not found: {args.corpus_path}")
    with open(args.corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    # truncate to keep test fast / focused
    text_in = (corpus_text or "").strip()
    if args.max_chars > 0 and len(text_in) > args.max_chars:
        text_in = text_in[: args.max_chars]

    client = DeepSeekHintClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        prompt_path=args.prompt_path,
        timeout=60.0,
    )

    prompt_preview = client.render_prompt(text_in)
    print("=== PROMPT PREVIEW (first 800 chars) ===")
    print(prompt_preview[:800])
    print("\n=== CALLING LLM ===")

    raw = client.call(text_in)

    print("\n=== RAW LLM OUTPUT ===")
    print(raw)

    ok, parsed, errs, saw_end = parse_plaintext_protocol(raw)
    # If END missing, try append once (common)
    if (not saw_end):
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
