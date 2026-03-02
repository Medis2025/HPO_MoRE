#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_more_backend.py  (STRICT, aligned with validation logic)

Key points:
- ORIGINAL text -> sentence split with ORIGINAL offsets
- ALWAYS call LLM any->EN once per sentence (LLM decides translate or keep)
- For each EN sentence:
    NER spans are extracted via TokenCRFWrapper.decode() + BIOES -> char spans (same as your validation)
    then HPOMoREInferer.infer(en_sentence) runs:
        encode_spans -> retrieve global HPO table -> margin gate -> optional LLM refine
- No "wrap whole sentence into one span" feature exists.

This backend requires backbone module exposing: HPOMoREInferer
"""

import os
import re
import sys
import time
import json
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

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
# LLM prompts (LLM decides translate OR keep)
# ================================
TRANSLATION_ANY2EN_PROMPT = """You are a medical translator.

Task:
Convert the following clinical sentence into professional English.

Mandatory rules:
- If the input is already fully English, output it EXACTLY unchanged (same wording, same punctuation).
- If the input contains any non-English (e.g., Chinese), translate it into natural, concise, professional English.
- If the input is mixed, translate ONLY the non-English parts, and keep existing English medical terms unchanged.
- Do NOT add, infer, or guess diagnoses.
- Preserve negations, dates, time expressions, severity, frequency, duration, and uncertainty markers exactly.
- Output ONLY the final English sentence. No explanations. No extra formatting.

Input sentence:
{{text}}
"""

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


# ================================
# LLM Translator (DeepSeek OpenAI-compat) — LLM-only, no fallbacks
# ================================
class LLMTranslator:
    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        timeout_sec: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 256,
        api_base: Optional[str] = None,
        api_base_env: str = "DEEPSEEK_API_BASE",
    ):
        self.provider = (provider or "deepseek").strip().lower()
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.timeout_sec = int(timeout_sec)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.api_base = (api_base or os.environ.get(api_base_env, "")).strip().rstrip("/")

        self._cache: Dict[Tuple[str, str], str] = {}

        if not self.api_key:
            raise RuntimeError("Missing LLM api_key (set env via --llm-api-key-env).")
        if not self.api_base:
            raise RuntimeError("Missing LLM api_base (set --llm-api-base or env DEEPSEEK_API_BASE).")
        if not self.model:
            raise RuntimeError("Missing LLM model (--llm-model).")

    def any2en_sentence(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        key = ("any2en", s)
        if key in self._cache:
            return self._cache[key]
        prompt = TRANSLATION_ANY2EN_PROMPT.replace("{{text}}", s)
        out = (self._call_llm(prompt) or "").strip()
        if not out:
            raise RuntimeError("LLM returned empty translation for any2en.")
        self._cache[key] = out
        return out

    def en2zh(self, text: str) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        key = ("en2zh", s)
        if key in self._cache:
            return self._cache[key]
        prompt = TRANSLATION_EN2ZH_PROMPT.replace("{{text}}", s)
        out = (self._call_llm(prompt) or "").strip()
        if not out:
            raise RuntimeError("LLM returned empty translation for en2zh.")
        self._cache[key] = out
        return out

    def _call_llm(self, prompt: str) -> str:
        import requests

        url = f"{self.api_base}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]


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

        # "def" in your inferer candidates is often "hpo_def" multi-line;
        # for UI we still translate it as a block.
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
# BIOES span decode helpers (copied in spirit from your validation)
# ================================
def spans_from_bioes(seq_ids: List[int], id2label: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    token spans in token index space (inclusive boundaries),
    matching your hpo_lora_hpoid.py evaluate_and_refine().
    """
    spans = []
    s = -1
    for i, y in enumerate(seq_ids):
        tag = id2label.get(int(y), "O")
        if tag == "O":
            if s != -1:
                s = -1
            continue
        if tag.startswith("S-"):
            spans.append((i, i))
        elif tag.startswith("B-"):
            s = i
        elif tag.startswith("E-"):
            if s != -1:
                spans.append((s, i))
                s = -1
    return spans

def token_span_to_char_span(token_span: Tuple[int, int], offsets: List[Tuple[int, int]]) -> Tuple[int, int]:
    i0, i1 = token_span
    a = offsets[i0][0]
    b = offsets[i1][1]
    return (int(a), int(b))

def postprocess_text_span(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^[\s\.,;:()\[\]\{\}]+", "", t)
    t = re.sub(r"[\s\.,;:()\[\]\{\}]+$", "", t)
    return t.strip()


# ================================
# Backbone runtime
#   - Loads HPOMoREInferer from backbone module
#   - Monkeypatches NER span extraction to use TokenCRFWrapper.decode (BIOES) exactly like validation
# ================================
class BackboneRuntime:
    def __init__(
        self,
        backbone_path: str,
        backbone_module: str,

        hpo_json: str,
        model_dir: str,      # NER LoRA dir
        backbone: str,       # HF base backbone dir
        span_ckpt: str,      # SpanProj ckpt used by inferer
        prompt_path: str,    # LLM refiner prompt
        out_dir: str,

        init_encoder_from: Optional[str] = None,
        batch_size: int = 32,
        max_len: int = 512,
        seed: int = 42,
        topk: int = 35,
        tau_low: float = 0.05,
        tau_high: float = 0.20,
        hpo_chunk_size: int = 512,

        # backbone refiner LLM
        bb_api_key_env: str = "DEEPSEEK_API_KEY",
        bb_base_url: str = "https://api.deepseek.com",
        bb_model: str = "deepseek-chat",

        show_head: int = 5,
    ):
        self.loaded = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
                f"Exports: {exports[:120]}{'...' if len(exports)>120 else ''}"
            )

        os.makedirs(out_dir, exist_ok=True)

        # ---- Load tokenizer + NER model (LoRA) ----
        # Your training script saves: model.base.save_pretrained(outdir) + tokenizer.save_pretrained(outdir)
        # Since model.base is a PEFT model, load via peft.
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from peft import PeftModel

        # tokenizer: prefer model_dir if it contains tokenizer files; else fallback to backbone
        tok_src = model_dir if os.path.isdir(model_dir) else backbone
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)

        base_tc = AutoModelForTokenClassification.from_pretrained(
            backbone,
            num_labels=5,  # O,B,I,E,S (same as your script)
        )
        # attach LoRA adapter
        try:
            base_tc = PeftModel.from_pretrained(base_tc, model_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load PEFT adapter from model_dir={model_dir}: {e}")

        # Wrap into TokenCRFWrapper (use decode exactly like validation)
        # Import TokenCRFWrapper from your hpo_lora_hpoid.py
        # NOTE: CRF weights were not saved in your checkpoint; CRF will only work if you separately persist them.
        try:
            from hpo_lora_hpoid import TokenCRFWrapper as TokenCRFWrapperTrain
        except Exception as e:
            raise RuntimeError(
                "Cannot import TokenCRFWrapper from hpo_lora_hpoid.py. "
                "Make sure backbone-path includes the directory containing that file. "
                f"Original error: {e}"
            )

        # We default use_crf=False here to avoid “phantom CRF” (since you didn’t save CRF state).
        model_tc = TokenCRFWrapperTrain(base_tc, num_labels=5, use_crf=False).to(self.device)
        model_tc.eval()

        # ---- Load ontology + cfg + SpanProj for inferer ----
        from train_hpoid_span_contrastive import HPOConfig, HPOOntology, SpanProj
        from hpo_llm_refiner import LLMAPIClient, HPOCandidateRefiner

        cfg = HPOConfig(
            backbone=backbone,
            init_encoder_from=init_encoder_from,
            model_dir=model_dir,
            hpo_json=hpo_json,
            max_len=max_len,
            batch_size=batch_size,
            stride=0,
            hpo_topk=topk,
        )

        ontology = HPOOntology(hpo_json)

        if not os.path.isfile(span_ckpt):
            raise FileNotFoundError(f"Span checkpoint not found: {span_ckpt}")
        ckpt = torch.load(span_ckpt, map_location="cpu")
        hidden_size = getattr(model_tc.base.config, "hidden_size", None) or getattr(model_tc.base.base_model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = model_tc.base.config.hidden_size
        span_dim = ckpt.get("cfg", {}).get("hpoid_dim", 256)

        span_proj = SpanProj(in_dim=hidden_size, out_dim=span_dim, dropout=0.0).to(self.device)
        span_proj.load_state_dict(ckpt["span_proj_state"])
        span_proj.eval()

        # ---- LLM refiner (same as your pipeline) ----
        api_key = os.environ.get(bb_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Env var {bb_api_key_env} is not set (backbone refiner).")
        llm_client = LLMAPIClient(
            api_key=api_key,
            base_url=bb_base_url,
            model=bb_model,
            timeout=60.0,
            prompt_path=prompt_path,
        )
        refiner = HPOCandidateRefiner(llm_client, max_candidates=int(topk))

        # ---- Instantiate inferer (builds GLOBAL HPO table once) ----
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
            refiner=refiner,
            enrich_candidates_for_llm=True,
        )

        # ---- Monkeypatch inferer NER extraction to EXACT validation logic (decode + BIOES) ----
        self._install_validation_style_ner_extractor(tokenizer=tokenizer, model_tc=model_tc, cfg=cfg)

        # ---- Print global table info ----
        z = getattr(self.inferer, "z_hpo", None)
        ids = getattr(self.inferer, "hpo_ids", None)

        print("\n==============================")
        print("[BackboneRuntime] GLOBAL HPO TABLE READY")
        print("  entries:", len(ids) if ids is not None else None)
        print("  tensor shape:", tuple(z.shape) if z is not None else None)
        if z is not None:
            print("  tensor device:", z.device)
            print("  tensor dtype :", z.dtype)
        if ids and show_head > 0:
            h = min(int(show_head), len(ids))
            print(f"  first_{h}_ids:", ids[:h])
            print(f"  last_{h}_ids :", ids[-h:])
        print("==============================\n")

        self.loaded = True

    def _install_validation_style_ner_extractor(self, tokenizer: Any, model_tc: Any, cfg: Any) -> None:
        """
        This matches your evaluate_and_refine logic in hpo_lora_hpoid.py:
          - tokenize with return_offsets_mapping
          - decode BIOES tags via model_tc.decode
          - spans_from_bioes -> token spans
          - token_span_to_char_span -> char spans
          - postprocess mention
        """

        # labels.json is saved by your training script in cfg.model_dir
        # if present, respect it.
        label_json = os.path.join(cfg.model_dir, "labels.json")
        if os.path.isfile(label_json):
            with open(label_json, "r", encoding="utf-8") as f:
                j = json.load(f)
            id2label = {int(k): v for k, v in (j.get("id2label") or {}).items()} if isinstance(j.get("id2label"), dict) else None
        else:
            id2label = None

        if not id2label:
            # fallback to the exact label order used in your script
            # build_label_map: ["O","B-PHENO","I-PHENO","E-PHENO","S-PHENO"]
            id2label = {0: "O", 1: "B-PHENO", 2: "I-PHENO", 3: "E-PHENO", 4: "S-PHENO"}

        def _ner_extract_spans_validation_style(text: str):
            text = text or ""
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                add_special_tokens=True,
                truncation=True,
                max_length=cfg.max_len,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            offsets = enc["offset_mapping"][0].tolist()  # List[[a,b],...]

            # decode token tags (BIOES)
            pred = model_tc.decode(input_ids=input_ids, attention_mask=attn)[0].detach().cpu().tolist()

            # IMPORTANT: keep offsets aligned to token indices; in your val code you drop specials by a==b
            # Here, we do the same by simply letting spans_from_bioes run on all tokens,
            # but only mapping to char where a!=b.
            # To mirror your val behavior strictly, we filter to valid positions and compress.
            valid_idx = [i for i, (a, b) in enumerate(offsets) if a != b]
            pred_valid = [pred[i] for i in valid_idx]
            offsets_valid = [tuple(offsets[i]) for i in valid_idx]

            tok_spans = spans_from_bioes(pred_valid, id2label)

            spans = []
            for (ti0, ti1) in tok_spans:
                c0, c1 = token_span_to_char_span((ti0, ti1), offsets_valid)
                mention = postprocess_text_span(text[c0:c1])
                if not mention:
                    continue
                spans.append((int(c0), int(c1), "PHENO", None))
            return spans

        # monkeypatch the inferer method
        self.inferer._ner_extract_spans = lambda text: [
            # map into NerSpan dataclass expected by inferer
            # (start,end,label,score)
            type("NerSpan", (), {"start": s, "end": e, "label": lab, "score": sc})
            for (s, e, lab, sc) in _ner_extract_spans_validation_style(text)
        ]

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

    def analyze_one_sentence(self, en_sent: str, topk: int, tau_low: float, tau_high: float) -> Dict[str, Any]:
        en_sent = (en_sent or "").strip()
        if not en_sent:
            return {"en": "", "entities": []}

        # override thresholds per request
        self.inferer.topk = int(topk)
        self.inferer.tau_low = float(tau_low)
        self.inferer.tau_high = float(tau_high)

        spans_out = self.inferer.infer(en_sent)  # List[SpanInfer]

        entities: List[Dict[str, Any]] = []
        for sp in spans_out:
            margin = float(getattr(sp, "dual_margin", 0.0) or 0.0)
            if margin >= float(tau_high):
                gate = "EASY"
            elif margin <= float(tau_low):
                gate = "HARD"
            else:
                gate = "MEDIUM"

            used_llm = bool(getattr(sp, "used_llm", False))
            final_source = "llm" if used_llm and gate != "EASY" else "dual"

            cands = []
            for cand in (getattr(sp, "candidates", None) or []):
                if isinstance(cand, dict):
                    cands.append(self._normalize_candidate_fields(cand))

            entities.append({
                "text": getattr(sp, "mention", "") or "",
                "label": getattr(sp, "label", "PHENO") or "PHENO",
                "c0": int(getattr(sp, "span", (0, 0))[0]),
                "c1": int(getattr(sp, "span", (0, 0))[1]),
                "candidates": cands,
                "dual_best_id": getattr(sp, "dual_best_id", None),
                "dual_margin": float(margin),
                "gate": gate,
                "final_hpo_id": getattr(sp, "pred_id", None),
                "final_source": final_source,
            })

        return {"en": en_sent, "entities": entities}


# ================================
# Flask server
# ================================
def create_app(demo_dir: str, translator: LLMTranslator, backbone: BackboneRuntime) -> Flask:
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
        t0 = time.time()
        payload = request.get_json(force=True, silent=False)

        text = (payload.get("text") or "").strip()
        topk = int(payload.get("topk", 35))
        tau_low = float(payload.get("tau_low", 0.05))
        tau_high = float(payload.get("tau_high", 0.20))
        return_candidate_zh = bool(payload.get("return_candidate_zh", False))
        debug_print = bool(payload.get("debug_print", False))

        if not text:
            return jsonify({"spans": [], "meta": {"error": "empty text"}}), 400

        spans_out: List[Dict[str, Any]] = []

        try:
            if debug_print:
                print("\n==============================")
                print("[REQ] topk:", topk, "tau_low:", tau_low, "tau_high:", tau_high,
                      "return_candidate_zh:", return_candidate_zh)
                print("[REQ] text:\n", text)
                print("==============================")

            # 1) ORIGINAL sentence split
            sents_src = split_sentences_with_offsets(text)

            # 2) ALWAYS any->EN per sentence
            en_sents: List[Dict[str, Any]] = []
            for s in sents_src:
                src = s["src"]
                en = translator.any2en_sentence(src)
                en_sents.append({"sid": int(s["sid"]), "en": en})
                if debug_print:
                    print(f"[TRANS] sid={s['sid']} c0={s['c0']} c1={s['c1']}")
                    print("  SRC:", src)
                    print("  EN :", en)
                    print("")

            # 3) Backbone per sentence (entity-only)
            sid2out: Dict[int, Dict[str, Any]] = {}
            for item in en_sents:
                sid = int(item["sid"])
                en_sent = (item.get("en") or "").strip()
                sid2out[sid] = backbone.analyze_one_sentence(
                    en_sent=en_sent,
                    topk=topk,
                    tau_low=tau_low,
                    tau_high=tau_high,
                )

            # 4) Build response aligned to ORIGINAL offsets
            for ss in sents_src:
                sid = int(ss["sid"])
                out_one = sid2out.get(sid, {"en": "", "entities": []})
                en_sent = out_one.get("en", "")
                entities = out_one.get("entities", []) or []

                entity_blocks = []
                for e in entities:
                    if not isinstance(e, dict):
                        continue
                    cands = e.get("candidates", []) or []
                    if return_candidate_zh:
                        cands = enrich_candidates_with_zh(cands, translator)

                    entity_blocks.append({
                        "text": e.get("text", ""),
                        "label": e.get("label", "PHENO"),
                        "c0": e.get("c0", None),
                        "c1": e.get("c1", None),
                        "candidates": cands,
                        "dual_best_id": e.get("dual_best_id", "-"),
                        "dual_margin": float(e.get("dual_margin", 0.0) or 0.0),
                        "gate": e.get("gate", "MEDIUM"),
                        "final_hpo_id": e.get("final_hpo_id", None),
                        "final_source": e.get("final_source", None),
                    })

                spans_out.append({
                    "type": "sentence",
                    "sid": sid,
                    "c0": ss["c0"], "c1": ss["c1"],     # ORIGINAL offsets
                    "src": ss["src"],                   # ORIGINAL sentence
                    "en": en_sent,                      # EN sentence
                    "entities_en": [{"text": eb["text"], "label": eb["label"], "c0": eb["c0"], "c1": eb["c1"]}
                                    for eb in entity_blocks],
                    "entity_blocks": entity_blocks,
                    "align": {"type": "sentence", "confidence": None},
                })

            dt = time.time() - t0
            total_entities = sum(len(s.get("entity_blocks") or []) for s in spans_out)
            llm_called = sum(
                1 for s in spans_out for e in (s.get("entity_blocks") or [])
                if e.get("final_source") == "llm"
            )

            return jsonify({
                "spans": spans_out,
                "meta": {
                    "topk": topk,
                    "tau_low": tau_low,
                    "tau_high": tau_high,
                    "return_candidate_zh": return_candidate_zh,
                    "latency_sec": round(dt, 4),
                    "backbone_loaded": bool(backbone.loaded),
                    "stats": {
                        "total_sentences": len(spans_out),
                        "total_entities": int(total_entities),
                        "llm_called": int(llm_called),
                    }
                }
            })

        except Exception as e:
            print("\n[ERROR] /api/analyze_full crashed:")
            traceback.print_exc()
            dt = time.time() - t0
            return jsonify({
                "spans": [],
                "meta": {
                    "error": str(e),
                    "latency_sec": round(dt, 4),
                    "backbone_loaded": bool(getattr(backbone, "loaded", False)),
                }
            }), 500

    return app


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
    ap.add_argument("--init_encoder_from", type=str, default=None)
    ap.add_argument("--span_ckpt", type=str, required=True)
    ap.add_argument("--prompt_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=35)
    ap.add_argument("--tau_low", type=float, default=0.05)
    ap.add_argument("--tau_high", type=float, default=0.20)
    ap.add_argument("--hpo_chunk_size", type=int, default=512)

    # backbone refiner LLM
    ap.add_argument("--bb-api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--bb-base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--bb-model", type=str, default="deepseek-chat")

    # translator LLM
    ap.add_argument("--llm-provider", type=str, default="deepseek")
    ap.add_argument("--llm-api-key-env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--llm-model", type=str, default="deepseek-chat")
    ap.add_argument("--llm-timeout-sec", type=int, default=60)
    ap.add_argument("--llm-temperature", type=float, default=0.0)
    ap.add_argument("--llm-max-tokens", type=int, default=256)
    ap.add_argument("--llm-api-base", type=str, default="")
    ap.add_argument("--llm-api-base-env", type=str, default="DEEPSEEK_API_BASE")

    args = ap.parse_args()

    demo_dir = os.path.abspath(args.demo_dir)
    if not os.path.isdir(demo_dir):
        raise FileNotFoundError(f"demo-dir not found: {demo_dir}")

    api_key = os.environ.get(args.llm_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Env var {args.llm_api_key_env} is not set (translator).")

    translator = LLMTranslator(
        provider=args.llm_provider,
        api_key=api_key,
        model=args.llm_model,
        timeout_sec=args.llm_timeout_sec,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        api_base=args.llm_api_base,
        api_base_env=args.llm_api_base_env,
    )

    backbone_rt = BackboneRuntime(
        backbone_path=args.backbone_path,
        backbone_module=args.backbone_module,

        hpo_json=args.hpo_json,
        model_dir=args.model_dir,
        backbone=args.backbone,
        span_ckpt=args.span_ckpt,
        prompt_path=args.prompt_path,
        out_dir=args.out_dir,
        init_encoder_from=args.init_encoder_from,

        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
        topk=args.topk,
        tau_low=args.tau_low,
        tau_high=args.tau_high,
        hpo_chunk_size=args.hpo_chunk_size,

        bb_api_key_env=args.bb_api_key_env,
        bb_base_url=args.bb_base_url,
        bb_model=args.bb_model,
    )

    app = create_app(demo_dir=demo_dir, translator=translator, backbone=backbone_rt)

    print(f"[server] demo-dir: {demo_dir}")
    print(f"[server] url: http://{args.host}:{args.port}/")
    print(f"[server] backbone_loaded: {backbone_rt.loaded}")
    print(f"[server] backbone_module: {args.backbone_module} (NER spans via decode+BIOES, aligned to validation)")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
