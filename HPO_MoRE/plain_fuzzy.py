#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_more_backend_fuzzy.py

✅ What you asked:
- If a single “sentence fragment” (already split by 。！？.!? etc.) is too long
  (≈ >100 tokens), further split it by comma-like separators (，, , ; ； : ： 、 etc.)
  *cutting at the last separator BEFORE the 100-token boundary*.

✅ Also keeps previous fix:
- One sentence implicit failure will NOT 500 the whole paragraph.
- Missing END sentinel is tolerated if SENT_EN exists (warn).

Notes on "100 tokens":
- True LLM tokens require a tokenizer; here we use a robust *approx token estimate*:
  - English: word/punct tokens
  - Chinese: treat each CJK char as ~1 token-ish, plus words/punct
This is good enough to prevent very long prompts.

Run is the same as your previous script.
"""

import os
import re
import json
import time
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from rapidfuzz import process, fuzz


# ================================
# small timing helper
# ================================
def now_s() -> float:
    return time.perf_counter()


# ==========================================================
# 1) Sentence split with ORIGINAL offsets + long-frag re-split
# ==========================================================
SENT_ANY_RE = re.compile(r"[^。！？；!?;\n.!?]+[。！？；!?;.!?]?|\n+", re.UNICODE)

# separators for secondary split (commas etc.)
SECOND_SPLIT_SEP_RE = re.compile(r"[，,；;：:、】【、()\[\]{}<>]|(?:\s+-\s+)", re.UNICODE)

# basic token-ish estimation: EN words + punct + CJK chars
CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]", re.UNICODE)
EN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def approx_tokens(s: str) -> int:
    """
    Approximate token count.
    - Count CJK chars as 1
    - Count English words as 1
    - Count punctuation as 1
    """
    if not s:
        return 0
    cjk = len(CJK_CHAR_RE.findall(s))
    enw = len(EN_WORD_RE.findall(s))
    punct = len(PUNCT_RE.findall(s))
    # avoid double-counting CJK as punct in most cases
    return int(cjk + enw + punct)


def _collect_sep_positions(text: str) -> List[int]:
    """Return separator positions (indices in text) that are good cut points."""
    pos = []
    for m in SECOND_SPLIT_SEP_RE.finditer(text):
        # cut *after* separator
        pos.append(m.end())
    # also allow newline cut points
    for m in re.finditer(r"\n+", text):
        pos.append(m.end())
    pos = sorted(set([p for p in pos if 0 < p < len(text)]))
    return pos


def split_long_fragment_by_seps(
    frag: str,
    *,
    max_tokens: int = 100,
    min_chunk_tokens: int = 20,
) -> List[Tuple[int, int]]:
    """
    Split a fragment into subranges [l,r) (relative to frag) so each chunk
    is roughly <= max_tokens, cutting at the last separator before boundary.
    If no separator is found, fall back to a hard cut near boundary.

    Returns list of (l, r) relative indices.
    """
    frag = frag or ""
    n = len(frag)
    if n == 0:
        return []

    if approx_tokens(frag) <= max_tokens:
        return [(0, n)]

    sep_pos = _collect_sep_positions(frag)
    out: List[Tuple[int, int]] = []

    i = 0
    # Greedy: grow to just under max_tokens, then cut at last sep before it.
    while i < n:
        # if remaining is short enough, take it
        if approx_tokens(frag[i:]) <= max_tokens:
            out.append((i, n))
            break

        # Find a tentative end by scanning forward until hitting token budget
        # We do a character scan to avoid expensive repeated tokenization
        # but still decide cut based on separators.
        j = i
        last_good = None

        # We will scan forward, and track token estimate incrementally.
        # For simplicity, we re-estimate on slices with increasing j but cap the loop.
        # Since max_tokens is small (100), this is fine in practice.
        # Soft cap: don't scan more than 600 chars ahead without cutting.
        scan_cap = min(n, i + 600)

        while j < scan_cap:
            j += 1
            if j in sep_pos:
                # candidate cut
                if approx_tokens(frag[i:j]) <= max_tokens:
                    last_good = j
                else:
                    break
            # stop if already over budget and we have some last_good
            if approx_tokens(frag[i:j]) > max_tokens and last_good is not None:
                break

        if last_good is None:
            # No separator within budget. Hard cut near the boundary.
            # Find smallest j such that tokens <= max_tokens, then cut there.
            j = i + 1
            while j < n and approx_tokens(frag[i:j]) <= max_tokens:
                j += 1
            j = max(i + 1, j - 1)

            # Avoid producing too tiny chunks: if chunk too small, extend a bit.
            if approx_tokens(frag[i:j]) < min_chunk_tokens and j < n:
                k = j
                while k < n and approx_tokens(frag[i:k]) < min_chunk_tokens:
                    k += 1
                j = min(n, k)

            out.append((i, j))
            i = j
            continue

        # Ensure chunk isn't too small (rare but can happen if sep clusters)
        if approx_tokens(frag[i:last_good]) < min_chunk_tokens and last_good < n:
            # try to extend to next sep after last_good within budget (if any)
            candidates = [p for p in sep_pos if p > last_good]
            extended = last_good
            for p in candidates:
                if approx_tokens(frag[i:p]) <= max_tokens:
                    extended = p
                else:
                    break
            last_good = extended

        out.append((i, last_good))
        i = last_good

    # Final cleanup: drop empty ranges
    out = [(l, r) for (l, r) in out if r > l and frag[l:r].strip() != ""]
    return out


def split_sentences_with_offsets(
    text: str,
    *,
    max_tokens_per_sent: int = 100,
    min_chunk_tokens: int = 20,
) -> List[Dict[str, Any]]:
    """
    First split by sentence punctuation using SENT_ANY_RE.
    Then for any fragment whose approx_tokens > max_tokens_per_sent,
    split again by comma-like separators before the boundary.
    Returns list of {sid, src, c0, c1} with ORIGINAL offsets.
    """
    out: List[Dict[str, Any]] = []
    sid = 0
    for m in SENT_ANY_RE.finditer(text or ""):
        frag = m.group(0)
        if not frag or frag.strip() == "":
            continue

        c0, c1 = m.start(), m.end()

        # secondary split for long fragments
        ranges = split_long_fragment_by_seps(
            frag,
            max_tokens=int(max_tokens_per_sent),
            min_chunk_tokens=int(min_chunk_tokens),
        )

        for (l, r) in ranges:
            sub = frag[l:r]
            if not sub or sub.strip() == "":
                continue
            out.append({"sid": sid, "src": sub, "c0": c0 + l, "c1": c0 + r})
            sid += 1

    return out


# ================================
# OpenAI-compat chat call (DeepSeek etc.)
# ================================
def call_llm_openai_compat(
    prompt: str,
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout: int = 60,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Tuple[str, float, Dict[str, Any]]:
    import requests

    api_base = (api_base or "").rstrip("/")
    if not api_base:
        raise RuntimeError("Missing api_base")
    if not api_key:
        raise RuntimeError("Missing api_key")
    if not model:
        raise RuntimeError("Missing model")

    url = f"{api_base}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    t0 = now_s()
    r = requests.post(url, headers=headers, json=payload, timeout=int(timeout))
    t1 = now_s()
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    return content, (t1 - t0), usage


# ================================
# Plaintext protocol parser (SENT_EN/HINT/END)
# ================================
_ALLOWED_POL = {"present", "absent", "uncertain", "unknown"}
_ALLOWED_TYPE_FALLBACK = "other"


def _norm_polarity(p: Any) -> str:
    s = str(p or "").strip().lower()
    if s in ("present", "pos", "positive", "yes", "y"):
        return "present"
    if s in ("absent", "neg", "negative", "no", "n", "denied", "without", "not_present"):
        return "absent"
    if s in ("uncertain", "possible", "suspected", "maybe", "probable", "unclear"):
        return "uncertain"
    if s in ("unknown", ""):
        return "uncertain"
    return "uncertain"


def parse_plaintext_protocol(raw: str) -> Tuple[bool, Dict[str, Any], List[str], bool]:
    """
    Expected:
      SENT_EN\t...
      HINT\t<hint_en>\t<polarity>\t<type>
      END

    ✅ Robustness:
    - If END missing but SENT_EN exists => ok (warn)
    """
    errs: List[str] = []
    if raw is None:
        return False, {}, ["raw is None"], False

    lines = [ln.rstrip("\n") for ln in (raw or "").splitlines() if ln.strip() != ""]
    if not lines:
        return False, {}, ["empty raw"], False

    sentence_en = ""
    hints: List[Dict[str, str]] = []
    saw_end = False

    for ln in lines:
        if ln.strip() == "END":
            saw_end = True
            break

        parts = ln.split("\t")
        if len(parts) == 1:
            parts = ln.split()

        tag = (parts[0] if parts else "").strip().upper()

        if ":" in ln and tag not in ("SENT_EN", "SENTENCE_EN", "EN", "HINT"):
            left, right = ln.split(":", 1)
            tag2 = left.strip().upper()
            if tag2 in ("SENT_EN", "SENTENCE_EN", "EN"):
                sentence_en = right.strip()
                continue

        if tag in ("SENT_EN", "SENTENCE_EN", "EN"):
            if len(parts) < 2:
                errs.append("SENT_EN missing value")
                continue
            val = "\t".join(parts[1:]).strip()
            if val:
                sentence_en = val
            else:
                errs.append("SENT_EN empty")
            continue

        if tag == "HINT":
            if len(parts) < 4:
                errs.append(f"HINT bad fields: {ln}")
                continue

            polarity = str(parts[-2]).strip().lower()
            tp = str(parts[-1]).strip().lower()
            hint_en = "\t".join(parts[1:-2]).strip()

            if not hint_en:
                errs.append(f"HINT empty hint_en: {ln}")
                continue

            if polarity not in _ALLOWED_POL:
                if polarity in ("pos", "positive", "yes", "y"):
                    polarity = "present"
                elif polarity in ("neg", "negative", "no", "n", "denied", "without", "not_present"):
                    polarity = "absent"
                elif polarity in ("unknown", ""):
                    polarity = "unknown"
                else:
                    errs.append(f"HINT invalid polarity: {ln}")
                    polarity = "unknown"

            if not tp:
                tp = _ALLOWED_TYPE_FALLBACK

            polarity = _norm_polarity(polarity)
            hints.append({"hint_en": hint_en, "polarity": polarity, "type": tp})
            continue

        errs.append(f"unknown line: {ln}")

    if not sentence_en.strip():
        errs.append("missing sentence_en")
    if not saw_end:
        errs.append("missing END sentinel")

    ok = (sentence_en.strip() != "")
    parsed = {"sentence_en": sentence_en.strip(), "hints": hints}
    return ok, parsed, errs, saw_end


# ================================
# Optional: candidate zh translation (unchanged)
# ================================
TRANSLATION_EN2ZH_PROMPT = """You are a medical translator.

Task:
Translate the following English medical term/sentence into concise, professional Chinese.

Mandatory rules:
- Do NOT add extra information.
- Preserve negations, uncertainty, and medical meaning.
- Output ONLY the Chinese translation. No explanations. No extra formatting.

English:
{{text}}
"""


class LLMTranslator:
    def __init__(
        self,
        *,
        api_key: str,
        api_base: str,
        model: str,
        timeout_sec: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        self.api_key = (api_key or "").strip()
        self.api_base = (api_base or "").strip().rstrip("/")
        self.model = (model or "").strip()
        self.timeout_sec = int(timeout_sec)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._cache: Dict[Tuple[str, str], str] = {}

        if not self.api_key:
            raise RuntimeError("Missing translator api_key")
        if not self.api_base:
            raise RuntimeError("Missing translator api_base")
        if not self.model:
            raise RuntimeError("Missing translator model")

    def en2zh(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        key = ("en2zh", s)
        if key in self._cache:
            return self._cache[key]
        prompt = TRANSLATION_EN2ZH_PROMPT.replace("{{text}}", s)
        out, _, _ = call_llm_openai_compat(
            prompt,
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout_sec,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        out = (out or "").strip()
        if not out:
            raise RuntimeError("LLM returned empty translation for en2zh.")
        self._cache[key] = out
        return out


def enrich_candidates_with_zh(cands: List[Dict[str, Any]], translator: LLMTranslator) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in (cands or []):
        if not isinstance(c, dict):
            continue
        cc = dict(c)

        name_en = cc.get("hpo_name") or cc.get("name") or ""
        if isinstance(name_en, list):
            name_en = name_en[0] if name_en else ""
        if isinstance(name_en, str) and name_en.strip():
            cc["name_zh"] = translator.en2zh(name_en)

        def_en = cc.get("hpo_def") or cc.get("def") or ""
        if isinstance(def_en, list):
            def_en = def_en[0] if def_en else ""
        if isinstance(def_en, str) and def_en.strip():
            cc["def_zh"] = translator.en2zh(def_en)

        syns = cc.get("hpo_synonyms") or cc.get("synonyms") or []
        if isinstance(syns, str):
            syns = [syns]
        if isinstance(syns, list) and syns:
            cc["synonyms_zh"] = [translator.en2zh(s) for s in syns if isinstance(s, str) and s.strip()]

        out.append(cc)
    return out


# ================================
# Implicit prompt runner (PLAINTEXT protocol)
# ================================
class ImplicitHintRunner:
    def __init__(
        self,
        *,
        prompt_path: str,
        api_key: str,
        api_base: str,
        model: str,
        timeout_sec: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 256,
        tolerate_missing_end: bool = True,
        auto_append_end_once: bool = True,
    ):
        self.prompt_path = os.path.abspath(prompt_path)
        if not os.path.isfile(self.prompt_path):
            raise FileNotFoundError(f"implicit prompt not found: {self.prompt_path}")

        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        self.api_key = (api_key or "").strip()
        self.api_base = (api_base or "").strip().rstrip("/")
        self.model = (model or "").strip()
        self.timeout_sec = int(timeout_sec)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.tolerate_missing_end = bool(tolerate_missing_end)
        self.auto_append_end_once = bool(auto_append_end_once)

        if not self.api_key:
            raise RuntimeError("Missing implicit LLM api_key")
        if not self.api_base:
            raise RuntimeError("Missing implicit LLM api_base")
        if not self.model:
            raise RuntimeError("Missing implicit LLM model")

    def run_one(self, sentence_src: str) -> Dict[str, Any]:
        s = (sentence_src or "").strip()
        if not s:
            return {
                "sentence_en": "",
                "hints": [],
                "raw": "",
                "latency_s": 0.0,
                "usage": {},
                "proto_ok": True,
                "proto_saw_end": True,
                "proto_errs": [],
            }

        prompt = self.prompt_template
        if "{{text}}" in prompt:
            prompt = prompt.replace("{{text}}", s)
        elif "{text}" in prompt:
            prompt = prompt.replace("{text}", s)
        elif "{SENT}" in prompt:
            prompt = prompt.replace("{SENT}", s)
        else:
            prompt = prompt.rstrip() + "\n\nInput sentence:\n" + s + "\n"

        raw, sec, usage = call_llm_openai_compat(
            prompt,
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout_sec,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        proto_ok, parsed, proto_errs, saw_end = parse_plaintext_protocol(raw)

        if (not saw_end) and self.auto_append_end_once:
            raw2 = (raw or "").rstrip() + "\nEND\n"
            proto_ok2, parsed2, proto_errs2, saw_end2 = parse_plaintext_protocol(raw2)
            if proto_ok2 and saw_end2:
                raw, parsed, proto_errs, saw_end, proto_ok = raw2, parsed2, proto_errs2, saw_end2, proto_ok2

        if not proto_ok:
            raise RuntimeError("implicit plaintext protocol invalid: " + " | ".join(proto_errs[:8]))

        if (not saw_end) and (not self.tolerate_missing_end):
            raise RuntimeError("implicit plaintext protocol invalid: missing END sentinel")

        sent_en = parsed.get("sentence_en", "")
        hints = parsed.get("hints", []) or []

        return {
            "sentence_en": sent_en,
            "hints": hints,
            "raw": raw,
            "latency_s": sec,
            "usage": usage,
            "proto_ok": True,
            "proto_saw_end": bool(saw_end),
            "proto_errs": proto_errs,
        }


# ================================
# HPO loader + fuzzy index (unchanged)
# ================================
def _as_first_str(x: Any) -> str:
    if isinstance(x, list):
        return str(x[0]) if x else ""
    if isinstance(x, str):
        return x
    return ""


def _as_list_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [str(v) for v in x if v is not None and str(v).strip()]
    return []


class HPOFuzzyIndex:
    def __init__(self, hpo_json_path: str):
        self.hpo_json_path = os.path.abspath(hpo_json_path)
        if not os.path.isfile(self.hpo_json_path):
            raise FileNotFoundError(f"hpo_json not found: {self.hpo_json_path}")

        with open(self.hpo_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            id2rec = data
        elif isinstance(data, list):
            id2rec = {}
            for it in data:
                if not isinstance(it, dict):
                    continue
                hid = it.get("id") or it.get("Id") or it.get("hpo_id")
                if hid:
                    id2rec[str(hid)] = it
        else:
            raise RuntimeError("Unsupported hpo_json structure (must be dict or list)")

        self.id2rec: Dict[str, Dict[str, Any]] = {}
        self.keys: List[str] = []
        self.key_meta: List[Tuple[str, str]] = []

        for hid, rec in id2rec.items():
            if not isinstance(rec, dict):
                continue
            hid = str(rec.get("Id") or rec.get("id") or hid).strip()
            if not hid:
                continue

            name = _as_first_str(rec.get("Name") or rec.get("name") or rec.get("label") or rec.get("preferred_label")) or hid
            syns = _as_list_str(rec.get("Synonym") or rec.get("synonym") or rec.get("synonyms"))

            self.id2rec[hid] = rec

            for k in [name] + syns:
                kk = str(k).strip()
                if not kk:
                    continue
                self.keys.append(kk)
                self.key_meta.append((hid, kk))

        if not self.keys:
            raise RuntimeError("HPOFuzzyIndex built empty keys list (no Name/Synonym).")

        self._cache: Dict[Tuple[str, int], List[Tuple[str, float, str]]] = {}

    def get_prompt_info(self, hpo_id: str, *, score: float) -> Dict[str, Any]:
        rec = self.id2rec.get(hpo_id, {}) or {}

        name = _as_first_str(rec.get("Name") or rec.get("name") or rec.get("label") or rec.get("preferred_label")) or hpo_id
        syns = _as_list_str(rec.get("Synonym") or rec.get("synonym") or rec.get("synonyms"))

        d = rec.get("Def") or rec.get("def") or ""
        orig_def = _as_first_str(d).strip()

        llm_def = _as_first_str(rec.get("llm_def") or "").strip()
        llm_add_def = _as_first_str(rec.get("llm_add_def") or "").strip()

        lines = [f"[HPO_ID] {hpo_id}", f"[NAME] {name}"]
        if syns:
            lines.append(f"[SYN] {'; '.join(syns)}")
        if orig_def:
            lines.append(f"[DEF] {orig_def}")
        if llm_def and llm_def != orig_def:
            lines.append(f"[LLM_DEF] {llm_def}")
        if llm_add_def:
            lines.append(f"[ADD_DEF] {llm_add_def}")

        return {
            "hpo_id": hpo_id,
            "hpo_name": name,
            "hpo_def": "\n".join(lines),
            "hpo_synonyms": syns,
            "hpo_orig_def": orig_def,
            "hpo_llm_def": llm_def,
            "hpo_add_def": llm_add_def,
            "score": float(score),
        }

    def fuzzy_topk(self, query: str, topk: int) -> List[Tuple[str, float, str]]:
        q = (query or "").strip()
        if not q:
            return []

        k = max(1, int(topk))
        cache_key = (q, k)
        if cache_key in self._cache:
            return self._cache[cache_key]

        raw = process.extract(
            q,
            self.keys,
            scorer=fuzz.token_set_ratio,
            limit=max(k * 8, 50),
        )

        best: Dict[str, Tuple[float, str]] = {}
        for matched_key, score, key_idx in raw:
            hid, key_text = self.key_meta[int(key_idx)]
            sc = float(score)
            prev = best.get(hid)
            if (prev is None) or (sc > prev[0]):
                best[hid] = (sc, key_text)

        items = [(hid, sc_key[0], sc_key[1]) for hid, sc_key in best.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        out = items[:k]
        self._cache[cache_key] = out
        return out


# ================================
# Fuzzy runtime + LLM refiner (unchanged)
# ================================
class FuzzyRuntime:
    def __init__(
        self,
        *,
        hpo_index: HPOFuzzyIndex,
        ref_prompt_path: str,
        ref_api_key: str,
        ref_api_base: str,
        ref_model: str,
        max_candidates: int = 35,
    ):
        from hpo_llm_refiner import LLMAPIClient, HPOCandidateRefiner

        self.hpo_index = hpo_index

        if not ref_api_key:
            raise RuntimeError("Missing refiner api_key")
        if not ref_api_base:
            raise RuntimeError("Missing refiner api_base")
        if not ref_model:
            raise RuntimeError("Missing refiner model")
        if not os.path.isfile(ref_prompt_path):
            raise FileNotFoundError(f"refiner prompt_path not found: {ref_prompt_path}")

        llm_client = LLMAPIClient(
            api_key=ref_api_key,
            base_url=ref_api_base,
            model=ref_model,
            timeout=60.0,
            prompt_path=ref_prompt_path,
        )
        self.refiner = HPOCandidateRefiner(llm_client, max_candidates=int(max_candidates))

    def infer_hint_entities(
        self,
        *,
        en_sent: str,
        hints: List[Dict[str, Any]],
        topk: int,
        tau_low: float,
        tau_high: float,
    ) -> List[Dict[str, Any]]:
        en_sent = (en_sent or "").strip()
        if not en_sent:
            return []

        entities: List[Dict[str, Any]] = []

        for h in (hints or []):
            if not isinstance(h, dict):
                continue

            pol = _norm_polarity(h.get("polarity"))
            if pol == "absent":
                continue

            hint_text = (h.get("hint_en") or h.get("text") or "").strip()
            if not hint_text:
                continue

            recalled = self.hpo_index.fuzzy_topk(hint_text, topk=int(topk))
            if not recalled:
                continue

            candidates: List[Dict[str, Any]] = []
            for hid, sc, key_text in recalled:
                info = self.hpo_index.get_prompt_info(hid, score=sc)
                info["matched_key"] = key_text
                candidates.append(info)

            best_score = float(candidates[0]["score"])
            second_score = float(candidates[1]["score"]) if len(candidates) > 1 else best_score
            margin = best_score - second_score

            if margin >= float(tau_high):
                gate = "EASY"
            elif margin <= float(tau_low):
                gate = "HARD"
            else:
                gate = "MEDIUM"

            entities.append(
                {
                    "text": hint_text,
                    "label": "HINT",
                    "c0": None,
                    "c1": None,
                    "span_type": "hint",
                    "hint_meta": h,
                    "candidates": candidates,
                    "dual_best_id": candidates[0]["hpo_id"],
                    "dual_best_score": best_score,
                    "dual_margin": float(margin),
                    "gate": gate,
                    "final_hpo_id": candidates[0]["hpo_id"],
                    "final_source": "dual",
                }
            )

        return entities


def refine_one_span(*, refiner: Any, context: str, mention: str, candidates: List[Dict[str, Any]]) -> List[int]:
    return refiner.refine(context, mention, candidates) or []


# ================================
# Flask server
# ================================
def create_app(
    *,
    demo_dir: str,
    implicit_runner: ImplicitHintRunner,
    fuzzy_rt: FuzzyRuntime,
    translator_for_candidates: Optional[LLMTranslator],
    translate_workers: int,
    refine_workers: int,
    max_tokens_per_sent: int,
    min_chunk_tokens: int,
) -> Flask:
    app = Flask(__name__, static_folder=None)
    CORS(app)

    @app.get("/")
    def index():
        return send_from_directory(demo_dir, "trans.html")

    @app.get("/demo/<path:filename>")
    def serve_demo(filename: str):
        return send_from_directory(demo_dir, filename)

    @app.post("/api/analyze_full")
    def analyze_full():
        t_all0 = now_s()
        payload = request.get_json(force=True, silent=False)

        text = (payload.get("text") or "").strip()
        topk = int(payload.get("topk", 35))
        tau_low = float(payload.get("tau_low", 15))
        tau_high = float(payload.get("tau_high", 25))
        return_candidate_zh = bool(payload.get("return_candidate_zh", False))
        debug_print = bool(payload.get("debug_print", False))

        # allow override per-request
        req_max_tokens = int(payload.get("max_tokens_per_sent", max_tokens_per_sent))
        req_min_chunk = int(payload.get("min_chunk_tokens", min_chunk_tokens))

        if not text:
            return jsonify({"spans": [], "meta": {"error": "empty text"}}), 400

        # ✅ split with secondary splitting for long fragments
        t0 = now_s()
        sents_src = split_sentences_with_offsets(
            text,
            max_tokens_per_sent=req_max_tokens,
            min_chunk_tokens=req_min_chunk,
        )
        t1 = now_s()

        n = len(sents_src)
        trans_workers = max(1, min(int(translate_workers), max(1, n)))
        implicit_out: List[Dict[str, Any]] = [
            {
                "sentence_en": "",
                "hints": [],
                "raw": "",
                "latency_s": 0.0,
                "usage": {},
                "ok": False,
                "error": "not_run",
                "proto_saw_end": False,
                "proto_errs": [],
            }
            for _ in range(n)
        ]

        if debug_print:
            print("\n==============================")
            print("[REQ] topk:", topk, "tau_low:", tau_low, "tau_high:", tau_high, "return_candidate_zh:", return_candidate_zh)
            print("[REQ] max_tokens_per_sent:", req_max_tokens, "min_chunk_tokens:", req_min_chunk)
            print("[REQ] text:\n", text)
            print("==============================")
            print(f"[STEP split] n_sent={len(sents_src)} time={t1-t0:.4f}s")

        # STEP-2: implicit per sentence (robust; no whole-500)
        t2 = now_s()
        with ThreadPoolExecutor(max_workers=trans_workers) as ex:
            futs = {}
            for s in sents_src:
                sid = int(s["sid"])
                futs[ex.submit(implicit_runner.run_one, s["src"])] = sid

            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    o = fut.result()
                    implicit_out[sid] = {
                        **o,
                        "ok": True,
                        "error": None,
                        "proto_saw_end": bool(o.get("proto_saw_end", False)),
                        "proto_errs": o.get("proto_errs", []) or [],
                    }
                except Exception as e:
                    implicit_out[sid] = {
                        "sentence_en": "",
                        "hints": [],
                        "raw": "",
                        "latency_s": 0.0,
                        "usage": {},
                        "ok": False,
                        "error": repr(e),
                        "proto_saw_end": False,
                        "proto_errs": [],
                    }

        t3 = now_s()
        if debug_print:
            print(f"[STEP implicit] workers={trans_workers} time={t3-t2:.4f}s")
            for s in sents_src:
                sid = int(s["sid"])
                o = implicit_out[sid]
                print(f"[IMPLICIT] sid={sid} ok={o.get('ok')} saw_end={o.get('proto_saw_end')} tok≈{approx_tokens(s['src'])} c0={s['c0']} c1={s['c1']}")
                if o.get("ok"):
                    print("  SRC:", s["src"])
                    print("  EN :", o.get("sentence_en", ""))
                    print("  hints:", [hh.get("hint_en") for hh in (o.get("hints") or []) if isinstance(hh, dict)])
                else:
                    print("  SRC:", s["src"])
                    print("  ERROR:", o.get("error"))
                print("")

        # STEP-3: fuzzy recall
        t4 = now_s()
        sid2_entities: Dict[int, List[Dict[str, Any]]] = {}
        for s in sents_src:
            sid = int(s["sid"])
            o = implicit_out[sid]
            if not o.get("ok"):
                sid2_entities[sid] = []
                continue
            en_sent = (o.get("sentence_en") or "").strip()
            hints = o.get("hints") or []
            sid2_entities[sid] = fuzzy_rt.infer_hint_entities(
                en_sent=en_sent, hints=hints, topk=topk, tau_low=tau_low, tau_high=tau_high
            )
        t5 = now_s()

        # STEP-4: refine
        t6 = now_s()
        refine_jobs = []
        for s in sents_src:
            sid = int(s["sid"])
            o = implicit_out[sid]
            if not o.get("ok"):
                continue

            en_sent = (o.get("sentence_en") or "").strip()
            hints = o.get("hints") or []

            hint_lines = []
            for h in hints:
                if isinstance(h, dict):
                    he = (h.get("hint_en") or "").strip()
                    pol = _norm_polarity(h.get("polarity"))
                    if he:
                        hint_lines.append(f"- {he} [{pol}]")
            hint_block = "\n".join(hint_lines).strip()
            context = f"EVIDENCE:\n{en_sent}\n\nHINTS:\n{hint_block if hint_block else '(none)'}"

            for ei, e in enumerate(sid2_entities.get(sid, [])):
                if e.get("gate") == "EASY":
                    continue
                mention = (e.get("text") or "").strip()
                cands = e.get("candidates") or []
                if mention and isinstance(cands, list) and cands:
                    refine_jobs.append((sid, ei, context, mention, cands))

        r_workers = max(1, min(int(refine_workers), max(1, len(refine_jobs)))) if refine_jobs else 0
        if refine_jobs:
            with ThreadPoolExecutor(max_workers=r_workers) as ex:
                futs = {}
                for (sid, ei, context, mention, cands) in refine_jobs:
                    futs[ex.submit(refine_one_span, refiner=fuzzy_rt.refiner, context=context, mention=mention, candidates=cands)] = (sid, ei)
                for fut in as_completed(futs):
                    sid, ei = futs[fut]
                    try:
                        idxs = fut.result()
                    except Exception:
                        continue
                    sid2_entities[sid][ei]["final_source"] = "llm"
                    if idxs and isinstance(idxs, list) and isinstance(idxs[0], int):
                        j0 = idxs[0]
                        cands = sid2_entities[sid][ei].get("candidates") or []
                        if 0 <= j0 < len(cands):
                            hid = cands[j0].get("hpo_id") or None
                            if hid:
                                sid2_entities[sid][ei]["final_hpo_id"] = hid

        t7 = now_s()

        # STEP-5: assemble
        t8 = now_s()
        spans_out: List[Dict[str, Any]] = []
        llm_called = 0
        total_entities = 0
        failed_sentences = 0
        missing_end_sentences = 0

        for s in sents_src:
            sid = int(s["sid"])
            o = implicit_out[sid]
            ok = bool(o.get("ok"))
            if not ok:
                failed_sentences += 1
            if ok and (o.get("proto_saw_end") is False):
                missing_end_sentences += 1

            en_sent = (o.get("sentence_en") or "").strip() if ok else ""
            hints = o.get("hints") or [] if ok else []
            entities = sid2_entities.get(sid, []) or []

            entity_blocks = []
            for e in entities:
                if not isinstance(e, dict):
                    continue
                cands = e.get("candidates", []) or []
                if return_candidate_zh:
                    if translator_for_candidates is None:
                        raise RuntimeError("return_candidate_zh=true but translator_for_candidates is None")
                    cands = enrich_candidates_with_zh(cands, translator_for_candidates)

                if e.get("final_source") == "llm":
                    llm_called += 1
                total_entities += 1

                entity_blocks.append(
                    {
                        "text": e.get("text", ""),
                        "label": e.get("label", "HINT"),
                        "c0": e.get("c0", None),
                        "c1": e.get("c1", None),
                        "span_type": e.get("span_type", "hint"),
                        "hint_meta": e.get("hint_meta", None),
                        "candidates": cands,
                        "dual_best_id": e.get("dual_best_id", "-"),
                        "dual_best_score": float(e.get("dual_best_score", 0.0) or 0.0),
                        "dual_margin": float(e.get("dual_margin", 0.0) or 0.0),
                        "gate": e.get("gate", "MEDIUM"),
                        "final_hpo_id": e.get("final_hpo_id", None),
                        "final_source": e.get("final_source", "dual"),
                    }
                )

            spans_out.append(
                {
                    "type": "sentence",
                    "sid": sid,
                    "c0": s["c0"],
                    "c1": s["c1"],
                    "src": s["src"],
                    "en": en_sent,
                    "hints": hints,
                    "entities_en": [{"text": eb["text"], "label": eb["label"], "c0": eb["c0"], "c1": eb["c1"]} for eb in entity_blocks],
                    "entity_blocks": entity_blocks,
                    "align": {"type": "sentence", "confidence": None},
                    "implicit_meta": {
                        "ok": ok,
                        "error": o.get("error"),
                        "latency_s": o.get("latency_s", 0.0),
                        "proto_saw_end": o.get("proto_saw_end", False),
                        "proto_errs": o.get("proto_errs", []) or [],
                    },
                    "split_meta": {
                        "approx_tokens_src": approx_tokens(s["src"]),
                        "max_tokens_per_sent": req_max_tokens,
                    },
                }
            )

        t9 = now_s()
        t_all1 = now_s()

        print(
            "[TIMING] split={:.4f}s implicit={:.4f}s fuzzy={:.4f}s refine={:.4f}s assemble={:.4f}s total={:.4f}s | sent={} ents={} refine_jobs={} failed_sent={} missing_end_sent={}".format(
                (t1 - t0),
                (t3 - t2),
                (t5 - t4),
                (t7 - t6),
                (t9 - t8),
                (t_all1 - t_all0),
                len(sents_src),
                total_entities,
                len(refine_jobs),
                failed_sentences,
                missing_end_sentences,
            )
        )

        return jsonify(
            {
                "spans": spans_out,
                "meta": {
                    "topk": topk,
                    "tau_low": tau_low,
                    "tau_high": tau_high,
                    "return_candidate_zh": return_candidate_zh,
                    "latency_sec": round((t_all1 - t_all0), 4),
                    "stats": {
                        "total_sentences": len(spans_out),
                        "failed_sentences": int(failed_sentences),
                        "missing_end_sentences": int(missing_end_sentences),
                        "total_entities": int(total_entities),
                        "llm_called": int(llm_called),
                        "refine_jobs": int(len(refine_jobs)),
                    },
                    "split_cfg": {
                        "max_tokens_per_sent": int(req_max_tokens),
                        "min_chunk_tokens": int(req_min_chunk),
                    },
                },
            }
        )

    return app


# ================================
# main
# ================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--demo-dir", type=str, required=True)

    ap.add_argument("--hpo_json", type=str, required=True)
    ap.add_argument("--prompt_path", type=str, required=True)          # refiner prompt
    ap.add_argument("--implicit-prompt", type=str, required=True)      # plaintext protocol prompt

    ap.add_argument("--topk", type=int, default=35)
    ap.add_argument("--tau_low", type=float, default=5.0)
    ap.add_argument("--tau_high", type=float, default=15.0)

    ap.add_argument("--llm-api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--llm-api-base-env", type=str, default="DEEPSEEK_API_BASE")
    ap.add_argument("--llm-api-base", type=str, default="")
    ap.add_argument("--llm-model", type=str, default="deepseek-chat")
    ap.add_argument("--llm-timeout-sec", type=int, default=60)
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=256)
    ap.add_argument("--translate-workers", type=int, default=8)

    ap.add_argument("--ref-api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--ref-api-base", type=str, default="https://api.deepseek.com")
    ap.add_argument("--ref-model", type=str, default="deepseek-chat")
    ap.add_argument("--refine-workers", type=int, default=8)

    ap.add_argument("--enable-candidate-zh", action="store_true")
    ap.add_argument("--zh-model", type=str, default="")
    ap.add_argument("--zh-max-tokens", type=int, default=256)

    # ✅ NEW: long sentence re-split config
    ap.add_argument("--max-tokens-per-sent", type=int, default=100, help="If src chunk > this (approx tokens), split further by commas.")
    ap.add_argument("--min-chunk-tokens", type=int, default=20, help="Avoid producing tiny chunks when splitting.")

    # protocol strictness toggles
    ap.add_argument("--strict_end", action="store_true", help="If set, require END sentinel strictly (otherwise tolerate).")
    ap.add_argument("--no_auto_append_end", action="store_true", help="Disable auto-append END fallback.")

    args = ap.parse_args()

    demo_dir = os.path.abspath(args.demo_dir)
    if not os.path.isdir(demo_dir):
        raise FileNotFoundError(f"demo-dir not found: {demo_dir}")

    llm_key = os.environ.get(args.llm_api_key_env, "").strip()
    if not llm_key:
        raise RuntimeError(f"Env var {args.llm_api_key_env} is not set (implicit LLM).")

    llm_base = (args.llm_api_base or os.environ.get(args.llm_api_base_env, "")).strip().rstrip("/")
    if not llm_base:
        raise RuntimeError(f"Missing implicit api_base: set --llm-api-base or env {args.llm_api_base_env}.")

    ref_key = os.environ.get(args.ref_api_key_env, "").strip()
    if not ref_key:
        raise RuntimeError(f"Env var {args.ref_api_key_env} is not set (refiner LLM).")

    hpo_index = HPOFuzzyIndex(args.hpo_json)
    print("[HPOFuzzyIndex] loaded ids:", len(hpo_index.id2rec), "keys:", len(hpo_index.keys))

    implicit_runner = ImplicitHintRunner(
        prompt_path=args.implicit_prompt,
        api_key=llm_key,
        api_base=llm_base,
        model=args.llm_model,
        timeout_sec=args.llm_timeout_sec,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        tolerate_missing_end=(not bool(args.strict_end)),
        auto_append_end_once=(not bool(args.no_auto_append_end)),
    )

    translator_for_candidates: Optional[LLMTranslator] = None
    if args.enable_candidate_zh:
        translator_for_candidates = LLMTranslator(
            api_key=llm_key,
            api_base=llm_base,
            model=(args.zh_model.strip() or args.llm_model),
            timeout_sec=args.llm_timeout_sec,
            temperature=args.llm_temperature,
            max_tokens=args.zh_max_tokens,
        )

    fuzzy_rt = FuzzyRuntime(
        hpo_index=hpo_index,
        ref_prompt_path=args.prompt_path,
        ref_api_key=ref_key,
        ref_api_base=args.ref_api_base,
        ref_model=args.ref_model,
        max_candidates=int(args.topk),
    )

    app = create_app(
        demo_dir=demo_dir,
        implicit_runner=implicit_runner,
        fuzzy_rt=fuzzy_rt,
        translator_for_candidates=translator_for_candidates,
        translate_workers=int(args.translate_workers),
        refine_workers=int(args.refine_workers),
        max_tokens_per_sent=int(args.max_tokens_per_sent),
        min_chunk_tokens=int(args.min_chunk_tokens),
    )

    print(f"[server] demo-dir: {demo_dir}")
    print(f"[server] url: http://{args.host}:{args.port}/")
    print(f"[server] implicit_prompt(plaintext): {os.path.abspath(args.implicit_prompt)}")
    print(f"[server] refine_prompt          : {os.path.abspath(args.prompt_path)}")
    print(f"[server] topk={int(args.topk)} tau_low={float(args.tau_low)} tau_high={float(args.tau_high)}")
    print(f"[server] translate_workers={int(args.translate_workers)} refine_workers={int(args.refine_workers)}")
    print(f"[server] enable_candidate_zh={bool(args.enable_candidate_zh)}")
    print(f"[server] max_tokens_per_sent={int(args.max_tokens_per_sent)} min_chunk_tokens={int(args.min_chunk_tokens)}")
    print(f"[server] strict_end={bool(args.strict_end)} auto_append_end={not bool(args.no_auto_append_end)}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()


"""
Run example:

python3 /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/plain_fuzzy.py \
  --host 0.0.0.0 --port 8008 \
  --demo-dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/demo \
  --hpo_json /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json \
  --prompt_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/prompts/select_hpo.txt \
  --implicit-prompt /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/Implicit_prop/implicit_span_proposal.txt \
  --topk 35 --tau_low 15 --tau_high 25 \
  --llm-model deepseek-chat --ref-api-base https://api.deepseek.com --ref-model deepseek-chat \
  --translate-workers 8 --refine-workers 8 \
  --max-tokens-per-sent 100 --min-chunk-tokens 20
"""
