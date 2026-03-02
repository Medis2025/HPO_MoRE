#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

# ====== your train-time components ======
# (same imports as your original pipeline)
from train_hpoid_span_contrastive import (
    HPOConfig,
    HPOOntology,
    TokenCRFWrapper,
    SpanProj,
    encode_spans,
    encode_hpo_gold_table,
)

# ====== optional LLM refiner ======
# (same interface as your original)
#   indices = refiner.refine(context, mention, candidates) -> List[int]
from hpo_llm_refiner import HPOCandidateRefiner


SPAN = Tuple[int, int]


@dataclass(frozen=True)
class NerSpan:
    start: int
    end: int
    label: str
    score: Optional[float] = None


@dataclass(frozen=True)
class SpanInfer:
    span: SPAN
    mention: str
    label: str
    dual_best_id: str
    dual_best_score: float
    dual_margin: float
    used_llm: bool
    pred_id: str
    candidates: List[Dict[str, Any]]  # topK candidates with score + optionally prompt fields


class HPOMoREInferer:
    """
    Strict end-to-end inference:

      raw_text
        -> NER spans (TokenCRFWrapper)
        -> DualLoRA span encoder retrieval against GLOBAL HPO table
        -> margin gate with tau_low / tau_high
        -> optional LLM refine on topK candidates
        -> final pred per span

    Design goals:
      - Build GLOBAL HPO table once at init (or via build_global_table()).
      - One single public method: infer(text) -> List[SpanInfer]
      - Minimal assumptions: you only need to implement NER span extraction in one place.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model_tc: TokenCRFWrapper,
        span_proj: SpanProj,
        ontology: HPOOntology,
        cfg: HPOConfig,
        device: torch.device,
        # retrieval
        topk: int = 35,
        hpo_chunk_size: int = 512,
        # margin gate
        tau_low: float = 0.05,
        tau_high: float = 0.20,
        # optional LLM refiner
        refiner: Optional[HPOCandidateRefiner] = None,
        # if True, enrich candidates with name/def/syn for LLM prompt
        enrich_candidates_for_llm: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model_tc = model_tc
        self.span_proj = span_proj
        self.ontology = ontology
        self.cfg = cfg
        self.device = device

        self.topk = int(topk)
        self.hpo_chunk_size = int(hpo_chunk_size)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)

        self.refiner = refiner
        self.enrich_candidates_for_llm = bool(enrich_candidates_for_llm)

        # global HPO table
        self.z_hpo: Optional[torch.Tensor] = None
        self.hpo_ids: Optional[List[str]] = None
        self.id2idx: Optional[Dict[str, int]] = None

        # set eval
        self.model_tc.to(self.device)
        self.model_tc.eval()
        self.span_proj.to(self.device)
        self.span_proj.eval()

        # build global table immediately (strict behavior)
        self.build_global_table()

    # ---------------------------
    # Global HPO table (once)
    # ---------------------------
    def build_global_table(self) -> None:
        all_hpo_ids = sorted(list(self.ontology.data.keys()))
        if not all_hpo_ids:
            raise RuntimeError("[HPOMoREInferer] ontology has no HPO ids.")

        z_chunks: List[torch.Tensor] = []
        ids_full: List[str] = []

        with torch.no_grad():
            for s in range(0, len(all_hpo_ids), self.hpo_chunk_size):
                chunk_ids = all_hpo_ids[s : s + self.hpo_chunk_size]
                z_chunk, valid_ids = encode_hpo_gold_table(
                    self.model_tc,
                    self.span_proj,
                    self.tokenizer,
                    self.ontology,
                    chunk_ids,
                    device=self.device,
                    max_len=self.cfg.max_len,
                )
                if z_chunk is None or z_chunk.numel() == 0:
                    continue
                z_chunks.append(z_chunk)
                ids_full.extend(valid_ids)

        if not z_chunks:
            raise RuntimeError("[HPOMoREInferer] failed to build global HPO table (empty).")

        z = torch.cat(z_chunks, dim=0)
        self.z_hpo = z
        self.hpo_ids = ids_full
        self.id2idx = {hid: i for i, hid in enumerate(ids_full)}

    # ---------------------------
    # Public API
    # ---------------------------
    @torch.no_grad()
    def infer(self, text: str) -> List[SpanInfer]:
        """
        Full inference over one input string.
        Returns span-level predictions.
        """
        if self.z_hpo is None or self.hpo_ids is None:
            raise RuntimeError("[HPOMoREInferer] global HPO table not built.")

        spans = self._ner_extract_spans(text)
        if not spans:
            return []

        # Build batch for encode_spans
        left_texts = [text] * len(spans)
        left_spans = [(sp.start, sp.end) for sp in spans]

        # 1) encode spans -> z_left
        z_left = encode_spans(
            self.model_tc,
            self.span_proj,
            self.tokenizer,
            left_texts,
            left_spans,
            self.device,
            self.cfg.max_len,
        )  # [B, D]

        # 2) retrieval against global HPO table
        sims = z_left @ self.z_hpo.t()  # [B, N]
        k = min(self.topk, sims.size(1))
        vals, idxs = torch.topk(sims, k=k, dim=-1)  # [B, K]

        vals = vals.detach().cpu()
        idxs = idxs.detach().cpu()

        out: List[SpanInfer] = []

        for i, sp in enumerate(spans):
            mention = self._safe_slice(text, (sp.start, sp.end))

            # build topK candidate list
            cand_list: List[Dict[str, Any]] = []
            for r in range(k):
                idx_hpo = int(idxs[i, r].item())
                score = float(vals[i, r].item())
                hid = self.hpo_ids[idx_hpo]

                if self.enrich_candidates_for_llm:
                    info = self._get_hpo_prompt_info(hid)
                    info["score"] = score
                    cand_list.append(info)
                else:
                    cand_list.append({"hpo_id": hid, "score": score})

            # dual best + margin
            dual_best_id = cand_list[0]["hpo_id"]
            dual_best_score = float(cand_list[0]["score"])
            second_score = float(cand_list[1]["score"]) if len(cand_list) > 1 else dual_best_score
            margin = dual_best_score - second_score

            # 3) tau gating + optional LLM refine
            pred_id, used_llm = self._gate_and_refine(
                context=text[:512] if len(text) > 512 else text,
                mention=mention,
                candidates=cand_list,
                dual_best_id=dual_best_id,
                margin=margin,
            )

            out.append(
                SpanInfer(
                    span=(sp.start, sp.end),
                    mention=mention,
                    label=sp.label,
                    dual_best_id=dual_best_id,
                    dual_best_score=dual_best_score,
                    dual_margin=float(margin),
                    used_llm=used_llm,
                    pred_id=pred_id,
                    candidates=cand_list,
                )
            )

        return out

    # ---------------------------
    # Core gate logic (infer-time)
    # ---------------------------
    def _gate_and_refine(
        self,
        *,
        context: str,
        mention: str,
        candidates: List[Dict[str, Any]],
        dual_best_id: str,
        margin: float,
    ) -> Tuple[str, bool]:
        """
        Infer-time (no gold):

          if margin >= tau_high: trust Dual (no LLM)
          else:
            call LLM (if available):
              if LLM valid: use LLM top
              else fallback Dual

        Additionally you can interpret margin <= tau_low as "hard",
        but infer-time action is still: call LLM if available.
        """
        # easy: high margin => trust dual
        if margin >= self.tau_high:
            return dual_best_id, False

        # no LLM => trust dual
        if self.refiner is None:
            return dual_best_id, False

        # hard/medium => call LLM (same action at infer-time)
        try:
            idxs = self.refiner.refine(context, mention, candidates) or []
        except Exception:
            idxs = []

        if not idxs:
            return dual_best_id, True

        pred_idx = idxs[0]
        if not isinstance(pred_idx, int) or pred_idx < 0 or pred_idx >= len(candidates):
            return dual_best_id, True

        llm_id = candidates[pred_idx].get("hpo_id") or dual_best_id
        return llm_id, True

    # ---------------------------
    # NER span extraction (YOU may need to adapt this)
    # ---------------------------
    def _ner_extract_spans(self, text: str) -> List[NerSpan]:
        """
        Extract mention spans from your NER model.

        You MUST adapt this to match your TokenCRFWrapper’s real API.
        Keep everything else unchanged.

        Common patterns I’ve seen:
          - model_tc.predict_spans(text, tokenizer, max_len=...) -> List[(s,e,label,score?)]
          - model_tc.predict(texts=[...]) -> per-token labels -> convert to spans

        Current implementation tries a few method names; if none exist, raises.
        """

        # --- 1) direct span API (preferred) ---
        for name in ["predict_spans", "extract_spans", "decode_spans", "ner_spans"]:
            fn = getattr(self.model_tc, name, None)
            if callable(fn):
                spans = fn(text, tokenizer=self.tokenizer, max_len=self.cfg.max_len)
                # normalize
                out: List[NerSpan] = []
                for item in spans:
                    # allow tuple/list/dict
                    if isinstance(item, dict):
                        s = int(item["start"])
                        e = int(item["end"])
                        lab = str(item.get("label", "MENTION"))
                        sc = item.get("score", None)
                        out.append(NerSpan(s, e, lab, float(sc) if sc is not None else None))
                    else:
                        # assume (start,end,label,score?) or (start,end,label)
                        s = int(item[0]); e = int(item[1]); lab = str(item[2]) if len(item) > 2 else "MENTION"
                        sc = float(item[3]) if len(item) > 3 else None
                        out.append(NerSpan(s, e, lab, sc))
                return self._dedup_and_clip(out, len(text))

        raise RuntimeError(
            "[HPOMoREInferer] Cannot extract NER spans: "
            "please implement _ner_extract_spans() to match your TokenCRFWrapper API."
        )

    # ---------------------------
    # HPO candidate enrichment for LLM prompt
    # ---------------------------
    def _get_hpo_prompt_info(self, hpo_id: str) -> Dict[str, Any]:
        """
        Same spirit as your original get_hpo_prompt_info:
          - reads Name/Syn/Def + llm_def/llm_add_def from ontology.data
          - composes a structured hpo_def string for LLM
        """
        hid = self.ontology.resolve_id(hpo_id)
        rec = self.ontology.data.get(hid, {}) or {}

        raw_name = rec.get("Name") or rec.get("name") or rec.get("label") or rec.get("preferred_label")
        if isinstance(raw_name, list):
            name = raw_name[0] if raw_name else hid
        elif isinstance(raw_name, str):
            name = raw_name
        else:
            name = hid

        syns = rec.get("Synonym") or rec.get("synonym") or []
        if isinstance(syns, str):
            syns = [syns]
        if not isinstance(syns, list):
            syns = []
        syns = [str(s) for s in syns if s]

        d = rec.get("Def") or rec.get("def") or ""
        if isinstance(d, list):
            d = d[0] if d else ""
        if not isinstance(d, str):
            d = ""
        orig_def = d.strip()

        llm_def = rec.get("llm_def") or ""
        llm_def = llm_def.strip() if isinstance(llm_def, str) else ""

        llm_add_def = rec.get("llm_add_def") or ""
        llm_add_def = llm_add_def.strip() if isinstance(llm_add_def, str) else ""

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

    # ---------------------------
    # Utility
    # ---------------------------
    @staticmethod
    def _safe_slice(text: str, span: SPAN) -> str:
        s, e = span
        s = max(0, min(int(s), len(text)))
        e = max(0, min(int(e), len(text)))
        if e < s:
            s, e = e, s
        return text[s:e]

    @staticmethod
    def _dedup_and_clip(spans: List[NerSpan], n: int) -> List[NerSpan]:
        # clip + dedup by (start,end,label)
        seen = set()
        out: List[NerSpan] = []
        for sp in spans:
            s = max(0, min(sp.start, n))
            e = max(0, min(sp.end, n))
            if e <= s:
                continue
            key = (s, e, sp.label)
            if key in seen:
                continue
            seen.add(key)
            out.append(NerSpan(s, e, sp.label, sp.score))
        # sort left-to-right
        out.sort(key=lambda x: (x.start, x.end))
        return out
