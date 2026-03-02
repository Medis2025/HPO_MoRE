#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_more_backend.py  (STRICT + HINT/TRANSLATION ADDED; CHANGED: use HINTS as spans, NOT NER)

ONLY change vs your previous version:
- STEP-3 is changed from: sentence_en -> NER(decode+BIOES) -> spans -> DualLoRA
- to: sentence_en + hints[] -> treat EACH hint_en as one span (0..len(hint_en)) -> DualLoRA retrieval

Everything else is kept:
- sentence split with ORIGINAL offsets
- ONE implicit LLM call per sentence (implicit_span_proposal.txt) -> sentence_en + hints
- margin gate EASY/MED/HARD
- backend threaded LLM refine (prompt_path) only for non-EASY
- prints timing per step
- no fallbacks
"""

import os
import re
import sys
import json
import time
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ================================
# small timing helper
# ================================
def now_s() -> float:
    return time.perf_counter()


# ================================
# Sentence split (keep ORIGINAL offsets)
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
    payload = {
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
# TokenCRFWrapper (ACKNOWLEDGED: your provided wrapper)
#   - IMPORTANT: forward has output_hidden, mapped to output_hidden_states
# ================================
class TokenCRFWrapper(nn.Module):
    def __init__(self, base_model: Any, num_labels: int, use_crf: bool):
        super().__init__()
        self.base = base_model
        self.num_labels = num_labels
        self.use_crf = False  # STRICT: do not enable CRF unless you truly saved CRF state

    def forward(self, input_ids, attention_mask, labels=None, output_hidden: bool = False):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=output_hidden,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1] if output_hidden else None
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        return {"loss": loss, "logits": logits, "hidden": hidden}

    @torch.no_grad()
    def decode(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits
        return logits.argmax(-1)


# ================================
# Candidate ZH enrichment (optional)
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

        name_en = cc.get("name") or cc.get("hpo_name") or ""
        if isinstance(name_en, list):
            name_en = name_en[0] if name_en else ""
        if isinstance(name_en, str) and name_en.strip():
            cc["name_zh"] = translator.en2zh(name_en)

        def_en = cc.get("def") or cc.get("hpo_def") or ""
        if isinstance(def_en, list):
            def_en = def_en[0] if def_en else ""
        if isinstance(def_en, str) and def_en.strip():
            cc["def_zh"] = translator.en2zh(def_en)

        syns = cc.get("synonyms") or cc.get("hpo_synonyms") or []
        if isinstance(syns, str):
            syns = [syns]
        if isinstance(syns, list) and syns:
            cc["synonyms_zh"] = [translator.en2zh(s) for s in syns if isinstance(s, str) and s.strip()]

        out.append(cc)
    return out


# ================================
# Implicit prompt runner (ONE call per sentence)
#   - uses implicit_span_proposal.txt
#   - must output JSON with at least "sentence_en"
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
# Backbone runtime:
#   - loads HPOMoREInferer from backbone module
#   - IMPORTANT: we DO NOT use NER now (changed step),
#               but we still load model_tc + span_proj + global HPO table
#   - refine is done in backend threads via HPOCandidateRefiner
# ================================
class BackboneRuntime:
    def __init__(
        self,
        *,
        backbone_path: str,
        backbone_module: str,
        hpo_json: str,
        model_dir: str,
        backbone: str,
        span_ckpt: str,
        topk: int,
        tau_low: float,
        tau_high: float,
        batch_size: int,
        max_len: int,
        seed: int,
        hpo_chunk_size: int,
        # refiner prompt + LLM config (used in backend refine)
        ref_prompt_path: str,
        ref_api_key: str,
        ref_api_base: str,
        ref_model: str,
        show_head: int = 5,
    ):
        self.loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

        if backbone_path and os.path.isdir(backbone_path) and backbone_path not in sys.path:
            sys.path.insert(0, backbone_path)

        try:
            mod = __import__(backbone_module)
        except Exception as e:
            raise RuntimeError(f"Failed to import backbone module={backbone_module} from {backbone_path}: {e}")

        InfererCls = getattr(mod, "HPOMoREInferer", None)
        if InfererCls is None:
            exports = sorted([k for k in vars(mod).keys() if not k.startswith("_")])
            raise RuntimeError(
                f"Backbone module {backbone_module} does not expose HPOMoREInferer.\n"
                f"Exports: {exports[:120]}{'...' if len(exports) > 120 else ''}"
            )

        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from peft import PeftModel
        from train_hpoid_span_contrastive import HPOConfig, HPOOntology, SpanProj
        from hpo_llm_refiner import LLMAPIClient, HPOCandidateRefiner

        tok_src = model_dir if os.path.isdir(model_dir) else backbone
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

        base_tc = AutoModelForTokenClassification.from_pretrained(backbone, num_labels=5)
        try:
            base_tc = PeftModel.from_pretrained(base_tc, model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load PEFT adapter from model_dir={model_dir}: {e}")

        model_tc = TokenCRFWrapper(base_tc, num_labels=5, use_crf=False).to(self.device)
        model_tc.eval()

        cfg = HPOConfig(
            backbone=backbone,
            init_encoder_from=None,
            model_dir=model_dir,
            hpo_json=hpo_json,
            max_len=int(max_len),
            batch_size=int(batch_size),
            stride=0,
            hpo_topk=int(topk),
        )
        ontology = HPOOntology(hpo_json)

        if not os.path.isfile(span_ckpt):
            raise FileNotFoundError(f"Span checkpoint not found: {span_ckpt}")
        ckpt = torch.load(span_ckpt, map_location="cpu")
        hidden_size = getattr(base_tc.config, "hidden_size", None)
        if hidden_size is None and hasattr(base_tc, "base_model") and hasattr(base_tc.base_model, "config"):
            hidden_size = getattr(base_tc.base_model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Cannot determine hidden_size from base model config.")
        span_dim = ckpt.get("cfg", {}).get("hpoid_dim", 256)

        span_proj = SpanProj(in_dim=int(hidden_size), out_dim=int(span_dim), dropout=0.0).to(self.device)
        span_proj.load_state_dict(ckpt["span_proj_state"])
        span_proj.eval()

        # inferer builds GLOBAL HPO table in __init__ (strict)
        self.inferer = InfererCls(
            tokenizer=tokenizer,
            model_tc=model_tc,
            span_proj=span_proj,
            ontology=ontology,
            cfg=cfg,
            device=self.device,
            topk=int(topk),
            hpo_chunk_size=int(hpo_chunk_size),
            tau_low=float(tau_low),
            tau_high=float(tau_high),
            refiner=None,  # refine in backend
            enrich_candidates_for_llm=True,
        )

        # external refiner (selection)
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
        self.refiner = HPOCandidateRefiner(llm_client, max_candidates=int(topk))

        # print global table info
        z = getattr(self.inferer, "z_hpo", None)
        ids = getattr(self.inferer, "hpo_ids", None)
        print("\n==============================")
        print("[BackboneRuntime] GLOBAL HPO TABLE READY")
        print("  entries:", len(ids) if ids is not None else None)
        print("  tensor shape:", tuple(z.shape) if z is not None else None)
        if z is not None:
            print("  tensor device:", z.device)
            print("  tensor dtype :", z.dtype)
        if ids and int(show_head) > 0:
            h = min(int(show_head), len(ids))
            print(f"  first_{h}_ids:", ids[:h])
            print(f"  last_{h}_ids :", ids[-h:])
        print("==============================\n")

        self.loaded = True

    @staticmethod
    def _normalize_candidate_fields(c: Dict[str, Any]) -> Dict[str, Any]:
        cc = dict(c)
        if "name" not in cc and "hpo_name" in cc:
            cc["name"] = cc["hpo_name"]
        if "def" not in cc and "hpo_def" in cc:
            cc["def"] = cc["hpo_def"]
        if "synonyms" not in cc and "hpo_synonyms" in cc:
            cc["synonyms"] = cc["hpo_synonyms"]
        return cc

    @torch.no_grad()
    def infer_hint_entities(
        self,
        *,
        en_sent: str,
        hints: List[Dict[str, Any]],
        topk: int,
        tau_low: float,
        tau_high: float,
    ) -> List[Dict[str, Any]]:
        """
        CHANGED behavior:
        - No NER.
        - Each hint_en is treated as one span over itself: (0, len(hint_en))
        - Retrieval is done against inferer.z_hpo
        - Gate is computed from margin (top1 - top2)
        - final_hpo_id defaults to dual_best_id (refine later in backend)
        """
        from train_hpoid_span_contrastive import encode_spans  # local import, strict dependency

        en_sent = (en_sent or "").strip()
        if not en_sent:
            return []

        # collect hint texts (EN)
        hint_texts: List[str] = []
        hint_meta: List[Dict[str, Any]] = []
        for h in (hints or []):
            if not isinstance(h, dict):
                continue
            he = (h.get("hint_en") or h.get("text") or "").strip()
            if not he:
                continue
            hint_texts.append(he)
            hint_meta.append(h)

        if not hint_texts:
            return []

        # thresholds
        self.inferer.topk = int(topk)
        self.inferer.tau_low = float(tau_low)
        self.inferer.tau_high = float(tau_high)

        # span inputs: each hint is its own left_text, left_span=(0,len)
        left_texts = hint_texts
        left_spans = [(0, len(t)) for t in hint_texts]

        z_left = encode_spans(
            self.inferer.model_tc,
            self.inferer.span_proj,
            self.inferer.tokenizer,
            left_texts,
            left_spans,
            self.device,
            self.inferer.cfg.max_len,
        )  # [B,D]

        z_hpo = self.inferer.z_hpo
        hpo_ids = self.inferer.hpo_ids
        if z_hpo is None or hpo_ids is None:
            raise RuntimeError("inferer global table missing (z_hpo / hpo_ids)")

        sims = z_left @ z_hpo.t()  # [B,N]
        k = min(int(topk), sims.size(1))
        vals, idxs = torch.topk(sims, k=k, dim=-1)

        vals = vals.detach().cpu()
        idxs = idxs.detach().cpu()

        entities: List[Dict[str, Any]] = []
        for i, hint_text in enumerate(hint_texts):
            # candidates
            cand_list: List[Dict[str, Any]] = []
            for r in range(k):
                idx_hpo = int(idxs[i, r].item())
                score = float(vals[i, r].item())
                hid = hpo_ids[idx_hpo]

                info = self.inferer._get_hpo_prompt_info(hid)  # backbone-provided enrichment
                info["score"] = score
                cand_list.append(self._normalize_candidate_fields(info))

            dual_best_id = cand_list[0].get("hpo_id")
            dual_best_score = float(cand_list[0].get("score", 0.0) or 0.0)
            second_score = float(cand_list[1].get("score", dual_best_score) or dual_best_score) if len(cand_list) > 1 else dual_best_score
            margin = dual_best_score - second_score

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
                    "c0": None,  # hints are not char spans inside en_sent unless your implicit prompt provides offsets
                    "c1": None,
                    "span_type": "hint",
                    "hint_meta": hint_meta[i],
                    "candidates": cand_list,
                    "dual_best_id": dual_best_id,
                    "dual_best_score": dual_best_score,
                    "dual_margin": float(margin),
                    "gate": gate,
                    "final_hpo_id": dual_best_id,
                    "final_source": "dual",
                }
            )

        return entities


# ================================
# Batched refine worker
# ================================
def refine_one_span(
    *,
    refiner: Any,
    context: str,
    mention: str,
    candidates: List[Dict[str, Any]],
) -> List[int]:
    return refiner.refine(context, mention, candidates) or []


# ================================
# Flask server
# ================================
def create_app(
    *,
    demo_dir: str,
    implicit_runner: ImplicitHintRunner,
    backbone: BackboneRuntime,
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
        tau_low = float(payload.get("tau_low", 0.05))
        tau_high = float(payload.get("tau_high", 0.20))
        return_candidate_zh = bool(payload.get("return_candidate_zh", False))
        debug_print = bool(payload.get("debug_print", False))

        if not text:
            return jsonify({"spans": [], "meta": {"error": "empty text"}}), 400

        try:
            # STEP-1: sentence split
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

            # STEP-2: implicit prompt (translation + hints) in parallel
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

            # STEP-3: CHANGED HERE — use HINTS as spans (no NER)
            t4 = now_s()
            sid2_entities: Dict[int, List[Dict[str, Any]]] = {}
            for s in sents_src:
                sid = int(s["sid"])
                o = implicit_out[sid]
                en_sent = (o.get("sentence_en") or "").strip()
                hints = o.get("hints") or []
                ents = backbone.infer_hint_entities(en_sent=en_sent, hints=hints, topk=topk, tau_low=tau_low, tau_high=tau_high)
                sid2_entities[sid] = ents
            t5 = now_s()
            if debug_print:
                total_ents = sum(len(v) for v in sid2_entities.values())
                print(f"[STEP backbone(HINTS)] total_entities={total_ents} time={t5-t4:.4f}s")

            # STEP-4: backend LLM refine in parallel (only non-EASY spans)
            # context uses BOTH evidence + hints
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
                            hint_lines.append(f"- {he} [{pol or 'unknown'}]")
                hint_block = "\n".join(hint_lines).strip()
                context = f"EVIDENCE:\n{en_sent}\n\nHINTS:\n{hint_block if hint_block else '(none)'}"

                for ei, e in enumerate(sid2_entities.get(sid, [])):
                    gate = e.get("gate", "MEDIUM")
                    if gate == "EASY":
                        continue
                    mention = (e.get("text") or "").strip()
                    cands = e.get("candidates") or []
                    if not mention or not isinstance(cands, list) or not cands:
                        continue
                    refine_jobs.append((sid, ei, context, mention, cands))

            r_workers = max(1, min(int(refine_workers), max(1, len(refine_jobs)))) if refine_jobs else 0

            if refine_jobs:
                with ThreadPoolExecutor(max_workers=r_workers) as ex:
                    futs = {}
                    for (sid, ei, context, mention, cands) in refine_jobs:
                        fut = ex.submit(refine_one_span, refiner=backbone.refiner, context=context, mention=mention, candidates=cands)
                        futs[fut] = (sid, ei)
                    for fut in as_completed(futs):
                        sid, ei = futs[fut]
                        idxs = fut.result()
                        sid2_entities[sid][ei]["final_source"] = "llm"
                        if idxs and isinstance(idxs, list) and isinstance(idxs[0], int):
                            j0 = idxs[0]
                            cands = sid2_entities[sid][ei].get("candidates") or []
                            if 0 <= j0 < len(cands):
                                hid = cands[j0].get("hpo_id") or cands[j0].get("id") or None
                                if hid:
                                    sid2_entities[sid][ei]["final_hpo_id"] = hid

            t7 = now_s()
            if debug_print:
                print(f"[STEP refine] jobs={len(refine_jobs)} workers={r_workers} time={t7-t6:.4f}s")

            # STEP-5: assemble response (optionally translate candidates to zh)
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
                            "final_source": e.get("final_source", None),
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
                    }
                )

            t9 = now_s()
            if debug_print:
                print(f"[STEP assemble] time={t9-t8:.4f}s")

            t_all1 = now_s()
            print(
                "[TIMING] split={:.4f}s implicit={:.4f}s backbone(HINTS)={:.4f}s refine={:.4f}s assemble={:.4f}s total={:.4f}s | sent={} ents={} refine_jobs={}".format(
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
                        "backbone_loaded": bool(backbone.loaded),
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
                            "backbone_loaded": bool(getattr(backbone, "loaded", False)),
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

    ap.add_argument("--backbone-path", type=str, required=True)
    ap.add_argument("--backbone-module", type=str, default="hpo_MoRE_backbone")

    ap.add_argument("--hpo_json", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, required=True)
    ap.add_argument("--span_ckpt", type=str, required=True)

    # selection/refine prompt (HPOCandidateRefiner)
    ap.add_argument("--prompt_path", type=str, required=True)

    # implicit prompt (translation + hint)
    ap.add_argument(
        "--implicit-prompt",
        type=str,
        required=True,
        help="Path to implicit_span_proposal.txt (returns sentence_en + hints).",
    )

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=35)
    ap.add_argument("--tau_low", type=float, default=0.05)
    ap.add_argument("--tau_high", type=float, default=0.20)
    ap.add_argument("--hpo_chunk_size", type=int, default=512)

    # implicit LLM (one call per sentence)
    ap.add_argument("--llm-api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--llm-api-base-env", type=str, default="DEEPSEEK_API_BASE")
    ap.add_argument("--llm-api-base", type=str, default="", help="Override base; else use env")
    ap.add_argument("--llm-model", type=str, default="deepseek-chat")
    ap.add_argument("--llm-timeout-sec", type=int, default=60)
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=512)
    ap.add_argument("--translate-workers", type=int, default=8)

    # refiner LLM (span selection) - processed in backend threads
    ap.add_argument("--ref-api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--ref-api-base", type=str, default="https://api.deepseek.com")
    ap.add_argument("--ref-model", type=str, default="deepseek-chat")
    ap.add_argument("--refine-workers", type=int, default=8)

    # optional: candidate zh translation uses same (implicit) llm unless overridden
    ap.add_argument("--enable-candidate-zh", action="store_true", help="If set, allow return_candidate_zh=true.")
    ap.add_argument("--zh-model", type=str, default="", help="If empty, reuse --llm-model")
    ap.add_argument("--zh-max-tokens", type=int, default=256)

    args = ap.parse_args()

    demo_dir = os.path.abspath(args.demo_dir)
    if not os.path.isdir(demo_dir):
        raise FileNotFoundError(f"demo-dir not found: {demo_dir}")

    # implicit llm creds
    llm_key = os.environ.get(args.llm_api_key_env, "").strip()
    if not llm_key:
        raise RuntimeError(f"Env var {args.llm_api_key_env} is not set (implicit LLM).")

    llm_base = (args.llm_api_base or os.environ.get(args.llm_api_base_env, "")).strip().rstrip("/")
    if not llm_base:
        raise RuntimeError(f"Missing implicit api_base: set --llm-api-base or env {args.llm_api_base_env}.")

    # refiner creds
    ref_key = os.environ.get(args.ref_api_key_env, "").strip()
    if not ref_key:
        raise RuntimeError(f"Env var {args.ref_api_key_env} is not set (refiner LLM).")

    implicit_runner = ImplicitHintRunner(
        prompt_path=args.implicit_prompt,
        api_key=llm_key,
        api_base=llm_base,
        model=args.llm_model,
        timeout_sec=args.llm_timeout_sec,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
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

    backbone_rt = BackboneRuntime(
        backbone_path=args.backbone_path,
        backbone_module=args.backbone_module,
        hpo_json=args.hpo_json,
        model_dir=args.model_dir,
        backbone=args.backbone,
        span_ckpt=args.span_ckpt,
        topk=args.topk,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
        hpo_chunk_size=args.hpo_chunk_size,
        ref_prompt_path=args.prompt_path,
        ref_api_key=ref_key,
        ref_api_base=args.ref_api_base,
        ref_model=args.ref_model,
    )

    app = create_app(
        demo_dir=demo_dir,
        implicit_runner=implicit_runner,
        backbone=backbone_rt,
        translator_for_candidates=translator_for_candidates,
        translate_workers=int(args.translate_workers),
        refine_workers=int(args.refine_workers),
    )

    print(f"[server] demo-dir: {demo_dir}")
    print(f"[server] url: http://{args.host}:{args.port}/")
    print(f"[server] backbone_loaded: {backbone_rt.loaded}")
    print(f"[server] backbone_module: {args.backbone_module} (HINT spans, no NER)")
    print(f"[server] implicit_prompt: {os.path.abspath(args.implicit_prompt)}")
    print(f"[server] refine_prompt  : {os.path.abspath(args.prompt_path)}")
    print(f"[server] translate_workers={int(args.translate_workers)} refine_workers={int(args.refine_workers)}")
    print(f"[server] enable_candidate_zh={bool(args.enable_candidate_zh)}")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
