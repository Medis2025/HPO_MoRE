#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
com_fuzzy_more.py  (MINIMIZED-CHANGE REVISION v1.1)
===================================================

CHANGE REQUEST:
- If DeepSeek HTTP/network error happens (DNS / timeout / non-200 / ConnectionError),
  DO NOT crash the whole run.
- Skip that sample's hint/refine call and keep going.

What changed (minimal):
1) DeepSeekPlainClient.call():
   - wrapped requests.post with retries
   - returns "" on failure (unless --strict_http is enabled)
2) step_a_hint_extract():
   - if raw=="" (call failed), returns ok=False and empty hints; keeps pipeline alive
3) step_c_gated_refine_one():
   - if LLM refine call fails / empty, fallback to dual_best_id (fuzzy top1) and continue
4) Added CLI switches:
   --http_retries (default 2)
   --http_backoff (default 1.2)
   --strict_http (default 0)  # if 1, behaves like before: raise and stop
   --skip_sleep_sec (default 0.0) # optional tiny sleep after failure

   
python /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/com_fuzzy_more.py \
  --eval_roots \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GeneReviews \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GSC+ \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/ID-68 \
  --hpo_json /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json \
  --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/VAL_HINT_FUZZY_REFINE \
  --api_key_env DEEPSEEK_API_KEY \
  --hint_prompt_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/Implicit_prop/en_hint_extract.txt \
  --refine_prompt_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/prompts/select_hpo.txt \
  --topk 35 \
  --num_workers 16 \
  --tau_low 5 \
  --tau_high 25

Everything else unchanged: metrics, markdown/json outputs, debug jsonl, etc.
"""

import os
import re
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.exceptions import RequestException
from tqdm import tqdm
from rapidfuzz import fuzz, process

# optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ====== import dataset / ontology (same as original project) ======
from train_hpoid_span_contrastive import (
    HPOOntology,
    HPOIDSpanPairDataset,
)

logger = logging.getLogger("HPO_HINT_FUZZY_REFINE")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)

SPAN = Tuple[int, int]

# -------------------------
# Plaintext protocol parser (HINT stage)
# -------------------------
_TAG_RE = re.compile(r"^\s*(SENT_EN|SENTENCE_EN|EN|HINT)\s*[:\t ]\s*(.*)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*END\s*\.?\s*$", re.IGNORECASE)
_HPOID_RE = re.compile(r"\bHP:\d{7}\b")
_INDEX_RE = re.compile(r"\bINDEX\s*[:=]\s*(-?\d+)\b", re.IGNORECASE)


def _norm_polarity(p: Any) -> str:
    s = str(p or "").strip().lower()
    if s in ("present", "pos", "positive", "yes", "y"):
        return "present"
    if s in ("absent", "neg", "negative", "no", "n", "denied", "without", "not_present"):
        return "absent"
    return "uncertain"


def _split_fields_after_tag(payload: str) -> List[str]:
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


def parse_refine_choice(raw: str, candidates: List[Dict[str, Any]]) -> Optional[int]:
    if not raw:
        return None

    m = _INDEX_RE.search(raw)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    m2 = _HPOID_RE.search(raw)
    if m2:
        hid = m2.group(0)
        for i, c in enumerate(candidates):
            if c.get("hpo_id") == hid:
                return i

    m3 = re.search(r"(-?\d+)", raw)
    if m3:
        try:
            return int(m3.group(1))
        except Exception:
            return None

    return None


# -------------------------
# Plain DeepSeek client (NOW resilient)
# -------------------------
class DeepSeekPlainClient:
    """
    Plain /chat/completions client using a prompt template loaded from txt.

    Resilience:
      - retries on DNS / connect / timeout / non-200
      - returns "" on failure unless strict_http=True
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        prompt_path: str,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        http_retries: int = 2,
        http_backoff: float = 1.2,
        strict_http: bool = False,
        skip_sleep_sec: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.http_retries = int(http_retries)
        self.http_backoff = float(http_backoff)
        self.strict_http = bool(strict_http)
        self.skip_sleep_sec = float(skip_sleep_sec)

        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"prompt_path not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def render_prompt(self, text: str, **extra) -> str:
        p = self.prompt_template
        p = p.replace("{text}", text or "")
        p = p.replace("{{text}}", text or "")
        p = p.replace("{{context}}", text or "")
        for k, v in (extra or {}).items():
            p = p.replace("{" + str(k) + "}", str(v))
            p = p.replace("{{" + str(k) + "}}", str(v))
        return p

    def call(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_err: Optional[str] = None
        attempts = max(0, self.http_retries) + 1

        for a in range(attempts):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code != 200:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    raise RuntimeError(last_err)

                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                last_err = str(e)
                # if strict: raise immediately (old behavior)
                if self.strict_http:
                    raise

                # non-strict: log and retry/backoff; after all attempts return ""
                if a < attempts - 1:
                    sleep_s = self.http_backoff * (2 ** a)
                    logger.warning("[HTTP] call failed (attempt %d/%d): %s ; retry in %.2fs",
                                   a + 1, attempts, last_err, sleep_s)
                    time.sleep(sleep_s)
                else:
                    logger.warning("[HTTP] call failed (final attempt %d/%d): %s ; returning empty string (skip sample)",
                                   a + 1, attempts, last_err)
                    if self.skip_sleep_sec > 0:
                        time.sleep(self.skip_sleep_sec)
                    return ""


# -------------------------------------------------------------------------
# Helper: extract HPO info for prompt (unchanged)
# -------------------------------------------------------------------------
def get_hpo_prompt_info(ontology: HPOOntology, hpo_id: str) -> Dict[str, Any]:
    hid = ontology.resolve_id(hpo_id)
    rec = ontology.data.get(hid, {}) or {}

    raw_name = rec.get("Name") or rec.get("name") or rec.get("label") or rec.get("preferred_label")
    if isinstance(raw_name, list):
        name = raw_name[0] if raw_name else hid
    elif isinstance(raw_name, str):
        name = raw_name
    else:
        name = getattr(ontology, "names", {}).get(hid, hid)

    syns = rec.get("Synonym") or rec.get("synonym") or []
    if isinstance(syns, str):
        syns = [syns]
    elif not isinstance(syns, list):
        syns = []
    syns = [str(s) for s in syns if s]

    d = rec.get("Def") or rec.get("def") or ""
    if isinstance(d, list):
        d = d[0] if d else ""
    if not isinstance(d, str):
        d = ""
    orig_def = d.strip()

    llm_def = rec.get("llm_def") or ""
    if not isinstance(llm_def, str):
        llm_def = ""
    llm_def = llm_def.strip()

    llm_add_def = rec.get("llm_add_def") or ""
    if not isinstance(llm_add_def, str):
        llm_add_def = ""
    llm_add_def = llm_add_def.strip()

    lines = [f"[HPO_ID] {hid}", f"[NAME] {name}"]
    if syns:
        lines.append(f"[SYN] {'; '.join(syns)}")
    if orig_def:
        lines.append(f"[DEF] {orig_def}")
    if llm_def and llm_def != orig_def:
        lines.append(f"[LLM_DEF] {llm_def}")
    if llm_add_def:
        lines.append(f"[ADD_DEF] {llm_add_def}")

    return {
        "hpo_id": hid,
        "hpo_name": name,
        "hpo_def": "\n".join(lines),
        "hpo_synonyms": syns,
        "hpo_orig_def": orig_def,
        "hpo_llm_def": llm_def,
        "hpo_add_def": llm_add_def,
    }


# -------------------------------------------------------------------------
# Fuzzy index
# -------------------------------------------------------------------------
def build_hpo_name_index(ontology: HPOOntology, use_syn: bool = False) -> Dict[str, Any]:
    choices: List[str] = []
    choice2hid: Dict[str, str] = {}

    for hid in sorted(list(ontology.data.keys())):
        info = get_hpo_prompt_info(ontology, hid)
        name = (info.get("hpo_name") or "").strip()
        if not name:
            continue

        if name not in choice2hid:
            choices.append(name)
            choice2hid[name] = hid

        if use_syn:
            for s in info.get("hpo_synonyms", []) or []:
                s = (s or "").strip()
                if s and (s not in choice2hid):
                    choices.append(s)
                    choice2hid[s] = hid

    logger.info(f"[Index] HPO fuzzy index built: {len(choices)} strings (use_syn={use_syn}).")
    return {"choices": choices, "choice2hid": choice2hid}


# -------------------------------------------------------------------------
# STEP A: Hint extraction (NOW skip-safe)
# -------------------------------------------------------------------------
def step_a_hint_extract(
    context_text: str,
    hint_client: DeepSeekPlainClient,
    max_context_chars: int,
    debug: bool = False,
) -> Dict[str, Any]:
    ctx = (context_text or "").strip()
    if max_context_chars > 0 and len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars]

    prompt = hint_client.render_prompt(ctx)
    raw = hint_client.call(prompt)

    # ---- CHANGE: if http failed -> raw=="" -> skip this sample's hint stage safely
    if not raw.strip():
        pack = {"ok": False, "sentence_en": "", "hints_present": [], "errs": ["http_call_failed_or_empty"], "raw": ""}
        if debug:
            logger.info("[DEBUG][A] Hint extract skipped (empty raw from client).")
        return pack

    ok, parsed, errs, saw_end = parse_plaintext_protocol(raw)
    if not saw_end:
        raw2 = (raw or "").rstrip() + "\nEND\n"
        ok2, parsed2, errs2, saw_end2 = parse_plaintext_protocol(raw2)
        if ok2 and saw_end2:
            ok, parsed, errs, saw_end = ok2, parsed2, errs2, saw_end2

    hints = parsed.get("hints", []) or []
    present = [h for h in hints if (h.get("polarity") == "present") and (h.get("hint_en") or "").strip()]

    pack = {
        "ok": bool(ok),
        "sentence_en": parsed.get("sentence_en", ""),
        "hints_present": present,
        "errs": errs,
        "raw": raw,
    }

    if debug:
        logger.info("[DEBUG][A] Hint extract: ok=%s, present_hints=%d, errs=%s",
                    pack["ok"], len(present), (errs[:3] if errs else []))
        if present:
            logger.info("[DEBUG][A] Present hints sample: %s", present[:3])
        raw_lines = (raw or "").splitlines()
        tail = "\n".join(raw_lines[-30:])
        logger.info("[DEBUG][A] Hint raw tail:\n%s", tail)

    return pack


# -------------------------------------------------------------------------
# STEP B: Fuzzy recall (unchanged)
# -------------------------------------------------------------------------
def step_b_fuzzy_recall(
    hint_strings: List[str],
    mention_fallback: str,
    index: Dict[str, Any],
    ontology: HPOOntology,
    topk: int,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], float]:
    choices = index["choices"]
    choice2hid = index["choice2hid"]

    qs = [h.strip() for h in (hint_strings or []) if h and h.strip()]
    if not qs and (mention_fallback or "").strip():
        qs = [mention_fallback.strip()]

    hid2score: Dict[str, float] = {}
    for q in qs:
        matches = process.extract(
            q,
            choices,
            scorer=fuzz.WRatio,
            limit=max(10, topk * 3),
        )
        for choice, score, _ in matches:
            hid = choice2hid.get(choice)
            if not hid:
                continue
            s = float(score) / 100.0
            if s > hid2score.get(hid, -1.0):
                hid2score[hid] = s

    cand_sorted = sorted(hid2score.items(), key=lambda x: x[1], reverse=True)[:topk]
    candidates: List[Dict[str, Any]] = []
    for hid, s in cand_sorted:
        info = get_hpo_prompt_info(ontology, hid)
        info["score"] = float(s)
        candidates.append(info)

    if candidates:
        best = float(candidates[0]["score"])
        second = float(candidates[1]["score"]) if len(candidates) > 1 else best
        margin = best - second
    else:
        margin = 0.0

    if debug:
        logger.info("[DEBUG][B] Fuzzy queries=%s", qs[:5])
        if candidates:
            logger.info("[DEBUG][B] Top candidates:")
            for i, c in enumerate(candidates[:min(5, len(candidates))]):
                logger.info("  #%d %s %s score=%.3f", i, c["hpo_id"], c["hpo_name"], c["score"])
            logger.info("[DEBUG][B] margin=%.4f (top1-top2)", margin)
        else:
            logger.info("[DEBUG][B] No candidates returned by fuzzy recall.")

    return candidates, float(margin)


# -------------------------------------------------------------------------
# STEP 1: build candidates + metrics (NOW hint HTTP-safe)
# -------------------------------------------------------------------------
def build_candidates_with_hint_fuzzy(
    dataset_name: str,
    ds: HPOIDSpanPairDataset,
    ontology: HPOOntology,
    index: Dict[str, Any],
    hint_client: DeepSeekPlainClient,
    topk: int = 15,
    hint_max_context_chars: int = 1200,
    debug_n: int = 0,
    debug_jsonl_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    total = 0
    top1_hits = 0
    recall_hits = 0
    samples_for_llm: List[Dict[str, Any]] = []

    dbg_f = None
    if debug_jsonl_path:
        os.makedirs(os.path.dirname(debug_jsonl_path), exist_ok=True)
        dbg_f = open(debug_jsonl_path, "w", encoding="utf-8")

    logger.info(f"[Hint+Fuzzy] Building top-{topk} candidates for dataset={dataset_name}...")
    for i, ex in enumerate(tqdm(ds, desc=f"[Hint+Fuzzy] {dataset_name} - spans", leave=False)):
        total += 1
        left_text = ex["left_text"]
        c0, c1 = ex["left_span"]
        gold = ex["hpo_id"]

        c0 = max(0, min(int(c0), len(left_text)))
        c1 = max(0, min(int(c1), len(left_text)))
        mention_text = left_text[c0:c1]
        context = left_text

        debug = (debug_n > 0 and i < debug_n)

        hint_pack = step_a_hint_extract(
            context_text=context,
            hint_client=hint_client,
            max_context_chars=hint_max_context_chars,
            debug=debug,
        )
        present_hints = hint_pack["hints_present"]
        hint_strings = [(h.get("hint_en") or "").strip() for h in present_hints if (h.get("hint_en") or "").strip()]

        candidates, margin = step_b_fuzzy_recall(
            hint_strings=hint_strings,
            mention_fallback=mention_text,
            index=index,
            ontology=ontology,
            topk=topk,
            debug=debug,
        )

        if candidates:
            if candidates[0]["hpo_id"] == gold:
                top1_hits += 1
            if any(c["hpo_id"] == gold for c in candidates):
                recall_hits += 1

        gold_in_topk = any(c["hpo_id"] == gold for c in candidates)
        if gold_in_topk and candidates:
            sample = {
                "dataset": dataset_name,
                "idx": i,
                "context": context[:512] if len(context) > 512 else context,
                "mention": mention_text,
                "gold_id": gold,
                "candidates": candidates,
                "dual_best_id": candidates[0]["hpo_id"],
                "dual_margin": float(margin),
                "hints": present_hints,
            }
            samples_for_llm.append(sample)

        if dbg_f and debug:
            dbg_obj = {
                "dataset": dataset_name,
                "idx": i,
                "gold_id": gold,
                "mention": mention_text,
                "hint_ok": hint_pack["ok"],
                "hint_errs": hint_pack["errs"][:5],
                "hints_present": present_hints[:10],
                "fuzzy_top": [
                    {"hpo_id": c["hpo_id"], "hpo_name": c["hpo_name"], "score": c["score"]}
                    for c in candidates[:10]
                ],
                "margin": margin,
                "gold_in_topk": bool(gold_in_topk),
            }
            dbg_f.write(json.dumps(dbg_obj, ensure_ascii=False) + "\n")

    if dbg_f:
        dbg_f.close()

    dual_top1 = top1_hits / max(1, total)
    dual_recallK = recall_hits / max(1, total)

    logger.info(
        f"[Hint+Fuzzy] dataset={dataset_name} top1={dual_top1:.4f}, "
        f"recall@{topk}={dual_recallK:.4f}, total_spans={total}, "
        f"LLM_samples={len(samples_for_llm)}"
    )

    metrics = {
        "dual_top1": float(dual_top1),
        "dual_recallK": float(dual_recallK),
        "dual_top1_hits": int(top1_hits),
        "dual_recallK_hits": int(recall_hits),
        "total_spans": int(total),
        "llm_samples": int(len(samples_for_llm)),
    }
    return samples_for_llm, metrics


# -------------------------------------------------------------------------
# STEP C: gating + refine (NOW skip-safe)
# -------------------------------------------------------------------------
def format_candidates_for_refine(candidates: List[Dict[str, Any]], max_n: int) -> str:
    lines = []
    for i, c in enumerate(candidates[:max_n]):
        lines.append(f"{i}. {c.get('hpo_id','')}\t{c.get('hpo_name','')}")
    return "\n".join(lines)


def format_hints_for_refine(hints: List[Dict[str, Any]], max_n: int = 12) -> str:
    if not hints:
        return "(none)"
    lines = []
    for h in hints[:max_n]:
        lines.append(f"- {h.get('hint_en','').strip()} [{h.get('polarity','uncertain')}]")
    return "\n".join(lines)


def step_c_gated_refine_one(
    sample: Dict[str, Any],
    refine_client: DeepSeekPlainClient,
    tau_low: float,
    tau_high: float,
    topk: int,
    debug: bool = False,
) -> Tuple[bool, bool, bool]:
    context = sample["context"]
    mention = sample["mention"]
    gold_id = sample["gold_id"]
    candidates = sample["candidates"]
    dual_best_id = sample.get("dual_best_id")
    margin = float(sample.get("dual_margin", 0.0))
    hints = sample.get("hints", []) or []

    if not candidates or not dual_best_id:
        return False, False, False

    if margin >= tau_high:
        if debug:
            logger.info("[DEBUG][C] EASY: margin=%.4f >= tau_high=%.4f -> trust fuzzy top1", margin, tau_high)
        pipeline_hit = (dual_best_id == gold_id)
        return pipeline_hit, False, False

    llm_called = True
    cand_block = format_candidates_for_refine(candidates, max_n=topk)
    hint_block = format_hints_for_refine(hints, max_n=12)

    prompt = refine_client.render_prompt(
        context,
        context=context,
        mention=mention,
        candidates=cand_block,
        hints=hint_block,
        topk=str(min(topk, len(candidates))),
    )

    if debug:
        band = "HARD" if margin <= tau_low else "MED"
        logger.info("[DEBUG][C] %s: margin=%.4f -> call LLM", band, margin)

    raw = refine_client.call(prompt)

    # ---- CHANGE: if call failed -> raw=="" => fallback to dual_best_id and continue
    if not raw.strip():
        if debug:
            logger.info("[DEBUG][C] refine skipped (empty raw). fallback to fuzzy top1.")
        pred_final = dual_best_id
        pipeline_hit = (pred_final == gold_id)
        return pipeline_hit, llm_called, False

    pred_idx = parse_refine_choice(raw, candidates)
    if debug:
        tail = "\n".join((raw or "").splitlines()[-30:])
        logger.info("[DEBUG][C] LLM raw tail:\n%s", tail)
        logger.info("[DEBUG][C] Parsed refine choice idx=%s", str(pred_idx))

    if pred_idx is None or pred_idx < 0 or pred_idx >= len(candidates):
        pred_final = dual_best_id
        pipeline_hit = (pred_final == gold_id)
        return pipeline_hit, llm_called, False

    llm_hid = candidates[pred_idx]["hpo_id"]
    llm_hit = (llm_hid == gold_id)

    if margin <= tau_low:
        pred_final = llm_hid
    else:
        pred_final = llm_hid if llm_hit else dual_best_id

    pipeline_hit = (pred_final == gold_id)
    return pipeline_hit, llm_called, llm_hit


def run_llm_refine_for_dataset(
    dataset_name: str,
    samples_for_llm: List[Dict[str, Any]],
    refine_client: DeepSeekPlainClient,
    num_workers: int = 16,
    tau_low: float = 0.05,
    tau_high: float = 0.20,
    topk: int = 15,
    debug_n: int = 0,
    debug_jsonl_path: Optional[str] = None,
) -> Dict[str, float]:
    if not samples_for_llm:
        logger.warning(f"[LLM] dataset={dataset_name} has no samples for LLM refine.")
        return {
            "pipeline_top1": 0.0,
            "pipeline_top1_hits": 0,
            "n_samples": 0,
            "llm_calls": 0,
            "llm_top1_hits": 0,
            "llm_top1_conditional": 0.0,
        }

    n = len(samples_for_llm)
    logger.info(f"[LLM] Starting margin-gated refine on dataset={dataset_name} with {n} samples...")

    dbg_f = None
    if debug_jsonl_path:
        os.makedirs(os.path.dirname(debug_jsonl_path), exist_ok=True)
        dbg_f = open(debug_jsonl_path, "a", encoding="utf-8")

    def _run_one(j: int, s: Dict[str, Any]) -> Tuple[bool, bool, bool, Dict[str, Any]]:
        debug = (debug_n > 0 and j < debug_n)
        pipeline_hit, llm_called, llm_hit = step_c_gated_refine_one(
            sample=s,
            refine_client=refine_client,
            tau_low=tau_low,
            tau_high=tau_high,
            topk=topk,
            debug=debug,
        )
        dbg = {}
        if debug:
            dbg = {
                "dataset": dataset_name,
                "idx": s.get("idx"),
                "gold_id": s.get("gold_id"),
                "dual_best_id": s.get("dual_best_id"),
                "margin": float(s.get("dual_margin", 0.0)),
                "llm_called": bool(llm_called),
                "llm_hit": bool(llm_hit),
                "pipeline_hit": bool(pipeline_hit),
            }
        return pipeline_hit, llm_called, llm_hit, dbg

    pipeline_hits = 0
    llm_calls = 0
    llm_hits = 0

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [pool.submit(_run_one, j, s) for j, s in enumerate(samples_for_llm)]
        for fut in tqdm(as_completed(futures), total=n, desc=f"[LLM] {dataset_name} - gated refine", leave=False):
            try:
                pipeline_hit, llm_called, llm_hit, dbg = fut.result()
            except Exception as e:
                logger.warning(f"[LLM] Future error: {e}")
                continue

            if pipeline_hit:
                pipeline_hits += 1
            if llm_called:
                llm_calls += 1
                if llm_hit:
                    llm_hits += 1

            if dbg_f and dbg:
                dbg_f.write(json.dumps(dbg, ensure_ascii=False) + "\n")

    if dbg_f:
        dbg_f.close()

    pipeline_top1 = pipeline_hits / max(1, n)
    llm_cond = (llm_hits / llm_calls) if llm_calls > 0 else 0.0

    logger.info(
        f"[LLM] dataset={dataset_name} pipeline_top1={pipeline_top1:.4f} "
        f"(pipeline_hits={pipeline_hits}/{n}), llm_calls={llm_calls}, "
        f"llm_cond_top1={llm_cond:.4f} (llm_hits={llm_hits})"
    )

    return {
        "pipeline_top1": float(pipeline_top1),
        "pipeline_top1_hits": int(pipeline_hits),
        "n_samples": int(n),
        "llm_calls": int(llm_calls),
        "llm_top1_hits": int(llm_hits),
        "llm_top1_conditional": float(llm_cond),
    }


# -------------------------------------------------------------------------
# Plotting helpers (unchanged)
# -------------------------------------------------------------------------
def plot_comparisons(results_summary: Dict[str, Dict[str, Any]], out_dir: str, topk: int) -> Dict[str, str]:
    if not HAS_MPL:
        logger.warning("[PLOT] matplotlib not available, skip plotting.")
        return {"top1": "", "recall_llm": ""}

    dataset_names = [k for k in results_summary.keys() if not k.startswith("_")]
    if not dataset_names:
        return {"top1": "", "recall_llm": ""}

    dataset_names = sorted(dataset_names)

    dual_top1_vals = []
    pipe_top1_vals = []
    recall_vals = []
    llm_cond_vals = []

    for ds in dataset_names:
        m = results_summary[ds]
        dual_top1_vals.append(float(m.get("dual_top1", 0.0)))
        pipe_top1_vals.append(float(m.get("pipeline_top1", 0.0)))
        recall_vals.append(float(m.get("dual_recallK", 0.0)))
        llm_cond_vals.append(float(m.get("llm_top1_conditional", 0.0)))

    x = list(range(len(dataset_names)))
    width = 0.35

    plt.style.use("ggplot")
    fig_w = max(6.0, 1.5 * len(dataset_names))

    plt.figure(figsize=(fig_w, 4.5))
    plt.bar([xi - width / 2 for xi in x], dual_top1_vals, width=width, label="Fuzzy Top-1")
    plt.bar([xi + width / 2 for xi in x], pipe_top1_vals, width=width, label="Pipeline Top-1")
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Top-1 Accuracy")
    plt.title("Top-1 Comparison: Fuzzy vs Pipeline (margin-gated)")
    plt.legend()
    plt.tight_layout()
    top1_path = os.path.join(out_dir, "top1_comparison.png")
    plt.savefig(top1_path, dpi=200)
    plt.close()

    plt.figure(figsize=(fig_w, 4.5))
    plt.bar([xi - width / 2 for xi in x], recall_vals, width=width, label=f"Fuzzy Recall@{topk}")
    plt.bar([xi + width / 2 for xi in x], llm_cond_vals, width=width, label="LLM Conditional Top-1")
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Ratio")
    plt.title(f"Recall@{topk} vs LLM Conditional Top-1")
    plt.legend()
    plt.tight_layout()
    recall_path = os.path.join(out_dir, "recall_vs_llm.png")
    plt.savefig(recall_path, dpi=200)
    plt.close()

    return {"top1": top1_path, "recall_llm": recall_path}


# -------------------------------------------------------------------------
# CLI & main (minimized-change + http flags)
# -------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="HINT→FUZZY(recall)→LLM(refine) eval for HPO-ID with margin-gated refine (HTTP-failure safe)."
    )
    ap.add_argument("--eval_roots", type=str, nargs="+", required=True)
    ap.add_argument("--val_root", type=str, default=None)
    ap.add_argument("--hpo_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--tau_low", type=float, default=0.05)
    ap.add_argument("--tau_high", type=float, default=0.20)

    # DeepSeek
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-chat")

    # prompts
    ap.add_argument("--hint_prompt_path", type=str, required=True)
    ap.add_argument("--refine_prompt_path", type=str, required=True)

    # token / timeout
    ap.add_argument("--hint_max_tokens", type=int, default=512)
    ap.add_argument("--refine_max_tokens", type=int, default=256)
    ap.add_argument("--hint_timeout", type=float, default=60.0)
    ap.add_argument("--refine_timeout", type=float, default=60.0)

    # hint context
    ap.add_argument("--hint_max_context_chars", type=int, default=1200)

    # fuzzy options
    ap.add_argument("--use_syn", type=int, default=0, help="0=name only, 1=name+synonyms")

    # debug
    ap.add_argument("--debug_n", type=int, default=0)

    # ---- NEW: http resilience ----
    ap.add_argument("--http_retries", type=int, default=2)
    ap.add_argument("--http_backoff", type=float, default=1.2)
    ap.add_argument("--strict_http", type=int, default=0, help="1 -> raise on HTTP error (old behavior).")
    ap.add_argument("--skip_sleep_sec", type=float, default=0.0)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ontology = HPOOntology(args.hpo_json)
    logger.info(f"Loaded HPO ontology with {len(ontology.data)} nodes from {args.hpo_json}.")

    index = build_hpo_name_index(ontology, use_syn=bool(args.use_syn))

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Env var {args.api_key_env} not set. export {args.api_key_env}=...")

    hint_client = DeepSeekPlainClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        prompt_path=args.hint_prompt_path,
        timeout=args.hint_timeout,
        temperature=0.0,
        max_tokens=args.hint_max_tokens,
        http_retries=args.http_retries,
        http_backoff=args.http_backoff,
        strict_http=bool(args.strict_http),
        skip_sleep_sec=args.skip_sleep_sec,
    )
    refine_client = DeepSeekPlainClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        prompt_path=args.refine_prompt_path,
        timeout=args.refine_timeout,
        temperature=0.0,
        max_tokens=args.refine_max_tokens,
        http_retries=args.http_retries,
        http_backoff=args.http_backoff,
        strict_http=bool(args.strict_http),
        skip_sleep_sec=args.skip_sleep_sec,
    )

    cfg_out = {"cli": vars(args)}
    cfg_path = os.path.join(args.out_dir, "hpo_revise_llm_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved config to {cfg_path}")

    results_summary: Dict[str, Dict[str, Any]] = {}

    total_spans_all = 0
    dual_top1_hits_all = 0
    dual_recall_hits_all = 0
    llm_samples_all = 0
    pipeline_hits_all = 0
    llm_calls_all = 0
    llm_top1_hits_all = 0

    eval_roots = list(args.eval_roots)
    if args.val_root:
        eval_roots.append(args.val_root)

    for root in eval_roots:
        dataset_name = os.path.basename(root.rstrip("/"))
        logger.info(f"==== Evaluating dataset: {dataset_name} ====")

        ds = HPOIDSpanPairDataset(
            roots=[root],
            ontology=ontology,
            max_context_chars=256,
            max_syn=3,
        )
        if len(ds) == 0:
            logger.warning(f"[Eval] dataset={dataset_name} has no examples, skip.")
            continue

        debug_jsonl_path = os.path.join(args.out_dir, f"{dataset_name}.debug.jsonl")

        samples_for_llm, dual_metrics = build_candidates_with_hint_fuzzy(
            dataset_name=dataset_name,
            ds=ds,
            ontology=ontology,
            index=index,
            hint_client=hint_client,
            topk=args.topk,
            hint_max_context_chars=args.hint_max_context_chars,
            debug_n=args.debug_n,
            debug_jsonl_path=debug_jsonl_path,
        )

        llm_metrics = run_llm_refine_for_dataset(
            dataset_name=dataset_name,
            samples_for_llm=samples_for_llm,
            refine_client=refine_client,
            num_workers=args.num_workers,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            topk=args.topk,
            debug_n=args.debug_n,
            debug_jsonl_path=debug_jsonl_path,
        )

        pipeline_top1 = (
            llm_metrics["pipeline_top1_hits"] / dual_metrics["total_spans"]
            if dual_metrics["total_spans"] > 0 else 0.0
        )

        results_summary[dataset_name] = {
            "dual_top1": dual_metrics["dual_top1"],
            "dual_recallK": dual_metrics["dual_recallK"],
            "dual_top1_hits": dual_metrics["dual_top1_hits"],
            "dual_recallK_hits": dual_metrics["dual_recallK_hits"],
            "total_spans": dual_metrics["total_spans"],
            "llm_samples": dual_metrics["llm_samples"],
            "pipeline_top1": float(pipeline_top1),
            "pipeline_top1_hits": llm_metrics["pipeline_top1_hits"],
            "llm_calls": llm_metrics["llm_calls"],
            "llm_top1_hits": llm_metrics["llm_top1_hits"],
            "llm_top1_conditional": llm_metrics["llm_top1_conditional"],
        }

        total_spans_all += dual_metrics["total_spans"]
        dual_top1_hits_all += dual_metrics["dual_top1_hits"]
        dual_recall_hits_all += dual_metrics["dual_recallK_hits"]
        llm_samples_all += dual_metrics["llm_samples"]
        pipeline_hits_all += llm_metrics["pipeline_top1_hits"]
        llm_calls_all += llm_metrics["llm_calls"]
        llm_top1_hits_all += llm_metrics["llm_top1_hits"]

        logger.info(f"[DEBUG] Wrote per-sample debug jsonl to: {debug_jsonl_path}")

    if total_spans_all > 0:
        global_dual_top1 = dual_top1_hits_all / total_spans_all
        global_dual_recallK = dual_recall_hits_all / total_spans_all
        global_pipeline_top1 = pipeline_hits_all / total_spans_all
    else:
        global_dual_top1 = 0.0
        global_dual_recallK = 0.0
        global_pipeline_top1 = 0.0

    global_llm_conditional = (llm_top1_hits_all / llm_calls_all) if llm_calls_all > 0 else 0.0

    results_summary["_GLOBAL"] = {
        "dual_top1": float(global_dual_top1),
        "dual_recallK": float(global_dual_recallK),
        "pipeline_top1": float(global_pipeline_top1),
        "llm_top1_conditional": float(global_llm_conditional),
        "total_spans": int(total_spans_all),
        "dual_top1_hits": int(dual_top1_hits_all),
        "dual_recallK_hits": int(dual_recall_hits_all),
        "llm_samples": int(llm_samples_all),
        "pipeline_top1_hits": int(pipeline_hits_all),
        "llm_calls": int(llm_calls_all),
        "llm_top1_hits": int(llm_top1_hits_all),
    }

    logger.info(
        "[GLOBAL] Fuzzy top1=%.4f, recall@%d=%.4f, LLM conditional top1=%.4f, Pipeline top1=%.4f (total_spans=%d)",
        global_dual_top1, args.topk, global_dual_recallK,
        global_llm_conditional, global_pipeline_top1, total_spans_all,
    )

    summary_json_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON summary to {summary_json_path}")

    plot_paths = plot_comparisons(results_summary, args.out_dir, topk=args.topk)

    md_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPO Hint → Fuzzy Recall → LLM Refine Evaluation (Margin-Gated Pipeline)\n\n")

        f.write("## Command\n\n```bash\n")
        f.write("python com_fuzzy_more.py \\\n")
        for k, v in vars(args).items():
            if isinstance(v, list):
                for item in v:
                    f.write(f"  --{k} {item} \\\n")
            else:
                f.write(f"  --{k} {v} \\\n")
        f.write("```\n\n")

        g = results_summary.get("_GLOBAL", {})
        f.write("## Global Metrics (All Datasets Combined)\n\n")
        f.write(f"- Total spans: **{g.get('total_spans', 0)}**\n")
        f.write("- Fuzzy Top-1: **{:.4f}**\n".format(g.get("dual_top1", 0.0)))
        f.write("- Fuzzy Recall@{}: **{:.4f}**\n".format(args.topk, g.get("dual_recallK", 0.0)))
        f.write("- LLM Conditional Top-1 (on called samples): **{:.4f}**\n".format(g.get("llm_top1_conditional", 0.0)))
        f.write("- **Full Pipeline Top-1**: **{:.4f}**\n\n".format(g.get("pipeline_top1", 0.0)))

        f.write("## Metrics per Dataset\n\n")
        f.write(
            "| Dataset | Fuzzy Top-1 | Fuzzy Recall@{} | Total spans | LLM samples | LLM calls | LLM Top-1 (cond.) | Pipeline Top-1 |\n".format(
                args.topk
            )
        )
        f.write("|---------|------------:|----------------:|------------:|------------:|----------:|------------------:|---------------:|\n")

        for ds_name, m in results_summary.items():
            if ds_name.startswith("_"):
                continue
            f.write(
                f"| {ds_name} | "
                f"{m['dual_top1']:.4f} | "
                f"{m['dual_recallK']:.4f} | "
                f"{m['total_spans']:>12} | "
                f"{m['llm_samples']:>12} | "
                f"{m['llm_calls']:>10} | "
                f"{m['llm_top1_conditional']:.4f} | "
                f"{m['pipeline_top1']:.4f} |\n"
            )

        f.write("\n## Plots\n\n")
        if plot_paths.get("top1"):
            f.write(f"![Top-1 Comparison]({os.path.basename(plot_paths['top1'])})\n\n")
        if plot_paths.get("recall_llm"):
            f.write(f"![Recall@{args.topk} vs LLM Conditional Top-1]({os.path.basename(plot_paths['recall_llm'])})\n\n")

        f.write("\n## Notes on HTTP Failures\n\n")
        f.write("- This script will **not** stop on network/DNS/HTTP errors.\n")
        f.write("- On a failed hint/refine call: it returns empty output and the sample falls back to fuzzy-only.\n")
        f.write("- Control behavior with: `--http_retries`, `--http_backoff`, `--strict_http 1`.\n")

    logger.info(f"Markdown summary saved to: {md_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
