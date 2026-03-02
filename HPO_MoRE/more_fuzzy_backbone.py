#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_more_backend_fuzzy.py  (HINT-ONLY FUZZY RECALL TOPK + LLM REFINE)

What it does (STRICT, no fallbacks):
- ORIGINAL text -> sentence split with ORIGINAL offsets
- For EACH sentence:
    (A) ONE LLM call using --implicit-prompt
        -> JSON with:
           - sentence_en (EN translation; if already EN keep exact)
           - hints (0..K) in English  (each hint must have hint_en ideally)
    (B) For each hint_en:
        - Fuzzy recall TopK over global HPO list (Name + Synonym) using RapidFuzz
        - Margin gate (EASY/MED/HARD) based on top1 - top2 fuzzy score
        - LLM refine (your --prompt_path) for non-EASY hints:
             refiner.refine(context, hint_en, candidates) -> indices
             pick candidates[idx0] as final

Returns:
- per-sentence blocks with: src, en, hints, entity_blocks
- each entity is a hint-span, not a NER mention span

Notes:
- HPO list loaded from --hpo_json (supports common dict/list styles)
- Candidate enrichment includes Name/Syn/Def/llm_def/llm_add_def fields if present
- LLM refine uses your existing hpo_llm_refiner.py interfaces:
    from hpo_llm_refiner import LLMAPIClient, HPOCandidateRefiner

✅ Revisions in this file (MINIMAL):
1) JSON MODE for implicit LLM call:
   - If provider supports OpenAI-compatible `response_format={"type":"json_object"}`,
     we enable it via --llm-json-mode.
2) If hint polarity is ABSENT (negated), DO NOT run fuzzy recall / DO NOT create entity_blocks for it.
   - The hint still remains in spans_out["hints"] for transparency.
3) Polarity normalization:
   - unknown -> uncertain
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


# ================================
# sentence split (keep ORIGINAL offsets)
# ================================
SENT_ANY_RE = re.compile(r"[^。！？；!?;\n.!?]+[。！？；!?;.!?]?|\n+", re.UNICODE)


def split_sentences_with_offsets(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sid = 0
    for m in SENT_ANY_RE.finditer(text or ""):
        frag = m.group(0)
        if not frag or frag.strip() == "":
            continue
        out.append({"sid": sid, "src": frag, "c0": m.start(), "c1": m.end()})
        sid += 1
    return out


# ================================
# JSON parse (strict-ish, but robust to extra text)
# ================================
def safe_json_load(s: str) -> Optional[dict]:
    s2 = (s or "").strip()
    try:
        return json.loads(s2)
    except Exception:
        pass
    l = s2.find("{")
    r = s2.rfind("}")
    if l >= 0 and r > l:
        try:
            return json.loads(s2[l : r + 1])
        except Exception:
            return None
    return None


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
    max_tokens: int = 512,
    json_mode: bool = False,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    json_mode:
      - If True, tries to request strict JSON output using:
          response_format={"type":"json_object"}
      - If provider ignores it, output may still drift; safe_json_load is kept.
    """
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
    # ✅ JSON mode (OpenAI compatible)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    t0 = now_s()
    r = requests.post(url, headers=headers, json=payload, timeout=int(timeout))
    t1 = now_s()
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    return content, (t1 - t0), usage


# ================================
# Optional: candidate zh translation
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
        json_mode: bool = False,  # not required for translation, but keep signature consistent
    ):
        self.api_key = (api_key or "").strip()
        self.api_base = (api_base or "").strip().rstrip("/")
        self.model = (model or "").strip()
        self.timeout_sec = int(timeout_sec)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.json_mode = bool(json_mode)
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
            json_mode=False,  # translation is plain text
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
# Implicit prompt runner (ONE call per sentence)
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
        max_tokens: int = 512,
        json_mode: bool = True,
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
        self.json_mode = bool(json_mode)

        if not self.api_key:
            raise RuntimeError("Missing implicit LLM api_key")
        if not self.api_base:
            raise RuntimeError("Missing implicit LLM api_base")
        if not self.model:
            raise RuntimeError("Missing implicit LLM model")

    def run_one(self, sentence_src: str) -> Dict[str, Any]:
        s = (sentence_src or "").strip()
        if not s:
            return {"sentence_en": "", "hints": [], "raw": "", "latency_s": 0.0, "usage": {}}

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
            json_mode=self.json_mode,  # ✅ JSON mode here
        )
        obj = safe_json_load(raw)
        if obj is None or not isinstance(obj, dict):
            raise RuntimeError("implicit prompt returned non-JSON or invalid JSON object")

        sent_en = obj.get("sentence_en", "")
        if not isinstance(sent_en, str) or not sent_en.strip():
            raise RuntimeError("implicit prompt JSON missing/invalid sentence_en")

        hints = obj.get("hints", [])
        if hints is None:
            hints = []
        if not isinstance(hints, list):
            raise RuntimeError("implicit prompt JSON invalid hints (must be list)")

        cleaned_hints: List[Dict[str, Any]] = []
        for h in hints:
            if isinstance(h, dict):
                cleaned_hints.append(h)
            elif isinstance(h, str) and h.strip():
                cleaned_hints.append({"hint_en": h.strip(), "polarity": "unknown"})

        return {"sentence_en": sent_en.strip(), "hints": cleaned_hints, "raw": raw, "latency_s": sec, "usage": usage}


# ================================
# HPO loader + fuzzy index
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
    """
    Build a rapidfuzz search index over:
      key = Name and Synonym strings
      value = hpo_id + canonical record
    """

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

            all_keys = [name] + syns
            for k in all_keys:
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
# Fuzzy runtime + LLM refiner
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

        def _norm_polarity(p: Any) -> str:
            s = str(p or "").strip().lower()
            if s in ("present", "pos", "positive", "yes", "y"):
                return "present"
            if s in ("absent", "neg", "negative", "no", "n", "denied", "without", "not_present"):
                return "absent"
            if s in ("uncertain", "possible", "suspected", "maybe", "probable", "unclear"):
                return "uncertain"
            if s in ("unknown", ""):
                # ✅ normalize legacy "unknown" -> "uncertain"
                return "uncertain"
            return s

        entities: List[Dict[str, Any]] = []

        for h in (hints or []):
            if not isinstance(h, dict):
                continue

            # ✅ skip negated hints
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


# ================================
# Refine worker
# ================================
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

        if not text:
            return jsonify({"spans": [], "meta": {"error": "empty text"}}), 400

        try:
            t0 = now_s()
            sents_src = split_sentences_with_offsets(text)
            t1 = now_s()

            if debug_print:
                print("\n==============================")
                print("[REQ] topk:", topk, "tau_low:", tau_low, "tau_high:", tau_high,
                      "return_candidate_zh:", return_candidate_zh)
                print("[REQ] text:\n", text)
                print("==============================")
                print(f"[STEP split] n_sent={len(sents_src)} time={t1-t0:.4f}s")

            # STEP-2: implicit prompt per sentence (parallel)
            t2 = now_s()
            n = len(sents_src)
            trans_workers = max(1, min(int(translate_workers), max(1, n)))
            implicit_out: List[Optional[Dict[str, Any]]] = [None] * n

            with ThreadPoolExecutor(max_workers=trans_workers) as ex:
                futs = {}
                for s in sents_src:
                    sid = int(s["sid"])
                    fut = ex.submit(implicit_runner.run_one, s["src"])
                    futs[fut] = sid
                for fut in as_completed(futs):
                    sid = futs[fut]
                    implicit_out[sid] = fut.result()

            t3 = now_s()
            if debug_print:
                print(f"[STEP implicit] workers={trans_workers} time={t3-t2:.4f}s")
                for s in sents_src:
                    sid = int(s["sid"])
                    o = implicit_out[sid]
                    print(f"[IMPLICIT] sid={sid} c0={s['c0']} c1={s['c1']}")
                    print("  SRC:", s["src"])
                    print("  EN :", o["sentence_en"])
                    print("  hints:", [hh.get("hint_en") for hh in (o.get("hints") or []) if isinstance(hh, dict)])
                    print("")

            # STEP-3: fuzzy recall over hints (CPU)
            t4 = now_s()
            sid2_entities: Dict[int, List[Dict[str, Any]]] = {}
            for s in sents_src:
                sid = int(s["sid"])
                o = implicit_out[sid]
                en_sent = (o.get("sentence_en") or "").strip()
                hints = o.get("hints") or []
                ents = fuzzy_rt.infer_hint_entities(en_sent=en_sent, hints=hints, topk=topk, tau_low=tau_low, tau_high=tau_high)
                sid2_entities[sid] = ents
            t5 = now_s()
            if debug_print:
                total_ents = sum(len(v) for v in sid2_entities.values())
                print(f"[STEP fuzzy] total_entities={total_ents} time={t5-t4:.4f}s")

            # STEP-4: LLM refine for non-EASY
            t6 = now_s()
            refine_jobs = []
            for s in sents_src:
                sid = int(s["sid"])
                o = implicit_out[sid]
                en_sent = (o.get("sentence_en") or "").strip()
                hints = o.get("hints") or []

                hint_lines = []
                for h in hints:
                    if isinstance(h, dict):
                        he = (h.get("hint_en") or "").strip()
                        pol = (h.get("polarity") or "").strip()
                        if he:
                            hint_lines.append(f"- {he} [{pol or 'uncertain'}]")
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
                        fut = ex.submit(refine_one_span, refiner=fuzzy_rt.refiner, context=context, mention=mention, candidates=cands)
                        futs[fut] = (sid, ei)
                    for fut in as_completed(futs):
                        sid, ei = futs[fut]
                        idxs = fut.result()
                        sid2_entities[sid][ei]["final_source"] = "llm"
                        if idxs and isinstance(idxs, list) and isinstance(idxs[0], int):
                            j0 = idxs[0]
                            cands = sid2_entities[sid][ei].get("candidates") or []
                            if 0 <= j0 < len(cands):
                                hid = cands[j0].get("hpo_id") or None
                                if hid:
                                    sid2_entities[sid][ei]["final_hpo_id"] = hid

            t7 = now_s()
            if debug_print:
                print(f"[STEP refine] jobs={len(refine_jobs)} workers={r_workers} time={t7-t6:.4f}s")

            # STEP-5: assemble output (optionally translate candidates to zh)
            t8 = now_s()
            spans_out: List[Dict[str, Any]] = []
            llm_called = 0
            total_entities = 0

            for s in sents_src:
                sid = int(s["sid"])
                o = implicit_out[sid]
                en_sent = (o.get("sentence_en") or "").strip()
                hints = o.get("hints") or []
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
                        "hints": hints,  # includes absent hints for transparency
                        "entities_en": [{"text": eb["text"], "label": eb["label"], "c0": eb["c0"], "c1": eb["c1"]} for eb in entity_blocks],
                        "entity_blocks": entity_blocks,
                        "align": {"type": "sentence", "confidence": None},
                    }
                )

            t9 = now_s()
            if debug_print:
                print(f"[STEP assemble] time={t9-t8:.4f}s")

            t_all1 = now_s()
            print(
                "[TIMING] split={:.4f}s implicit={:.4f}s fuzzy={:.4f}s refine={:.4f}s assemble={:.4f}s total={:.4f}s | sent={} ents={} refine_jobs={}".format(
                    (t1 - t0),
                    (t3 - t2),
                    (t5 - t4),
                    (t7 - t6),
                    (t9 - t8),
                    (t_all1 - t_all0),
                    len(sents_src),
                    total_entities,
                    len(refine_jobs),
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
                            "total_entities": int(total_entities),
                            "llm_called": int(llm_called),
                            "refine_jobs": int(len(refine_jobs)),
                        },
                    },
                }
            )

        except Exception as e:
            print("\n[ERROR] /api/analyze_full crashed:")
            traceback.print_exc()
            t_all1 = now_s()
            return (
                jsonify(
                    {
                        "spans": [],
                        "meta": {
                            "error": str(e),
                            "latency_sec": round((t_all1 - t_all0), 4),
                        },
                    }
                ),
                500,
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
    ap.add_argument("--prompt_path", type=str, required=True)
    ap.add_argument("--implicit-prompt", type=str, required=True)

    ap.add_argument("--topk", type=int, default=35)
    ap.add_argument("--tau_low", type=float, default=5.0)
    ap.add_argument("--tau_high", type=float, default=15.0)

    ap.add_argument("--llm-api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--llm-api-base-env", type=str, default="DEEPSEEK_API_BASE")
    ap.add_argument("--llm-api-base", type=str, default="")
    ap.add_argument("--llm-model", type=str, default="deepseek-chat")
    ap.add_argument("--llm-timeout-sec", type=int, default=60)
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=512)
    ap.add_argument("--translate-workers", type=int, default=8)

    # ✅ JSON mode toggle
    ap.add_argument(
        "--llm-json-mode",
        action="store_true",
        help='Enable OpenAI-compatible JSON mode: response_format={"type":"json_object"} for implicit prompt calls.',
    )

    ap.add_argument("--ref-api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--ref-api-base", type=str, default="https://api.deepseek.com")
    ap.add_argument("--ref-model", type=str, default="deepseek-chat")
    ap.add_argument("--refine-workers", type=int, default=8)

    ap.add_argument("--enable-candidate-zh", action="store_true")
    ap.add_argument("--zh-model", type=str, default="")
    ap.add_argument("--zh-max-tokens", type=int, default=256)

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
        json_mode=bool(args.llm_json_mode),  # ✅
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
            json_mode=False,
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
    )

    print(f"[server] demo-dir: {demo_dir}")
    print(f"[server] url: http://{args.host}:{args.port}/")
    print(f"[server] implicit_prompt: {os.path.abspath(args.implicit_prompt)}")
    print(f"[server] refine_prompt  : {os.path.abspath(args.prompt_path)}")
    print(f"[server] topk={int(args.topk)} tau_low={float(args.tau_low)} tau_high={float(args.tau_high)}")
    print(f"[server] translate_workers={int(args.translate_workers)} refine_workers={int(args.refine_workers)}")
    print(f"[server] enable_candidate_zh={bool(args.enable_candidate_zh)}")
    print(f"[server] llm_json_mode={bool(args.llm_json_mode)}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()


"""
python3 /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/more_fuzzy_backbone.py  
 --host 0.0.0.0   --port 8008   --demo-dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/demo  
   --hpo_json /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json  
     --prompt_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/prompts/select_hpo.txt  
       --implicit-prompt /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/Implicit_prop/implicit_span_proposal.txt 
           --topk 35   --tau_low 15   --tau_high 25   --llm-model deepseek-chat   --ref-api-base https://api.deepseek.com  
 --ref-model deepseek-chat   --translate-workers 8   --refine-workers 8 --llm-json-mode
"""