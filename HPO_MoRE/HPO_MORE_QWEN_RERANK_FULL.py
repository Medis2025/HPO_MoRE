#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPO_MoRE_QWEN_RERANK.py

DualLoRAEnc + LOCAL Qwen3-Reranker-4B refine eval for HPO-ID with margin-based gating,
using a GLOBAL HPO table and enriched HPO JSON (with llm_def / llm_add_def).

REVISION (ONLY WHAT CHANGED vs previous revised version you got from me):
✅ Add default --val_root to:
     /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/val
✅ Ensure validation root is actually evaluated as an additional dataset (same pipeline)
   - It is appended to eval_roots
   - It shows up in JSON + Markdown tables and is included in GLOBAL aggregation

python /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/HPO_MORE_QWEN_RERANK_FULL.py \
  --eval_roots \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GeneReviews \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GSC+ \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/ID-68 \
  --val_root /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/val \
  --hpo_json /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json \
  --model_dir /cluster/home/gw/Backend_project/NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --init_encoder_from /cluster/home/gw/Backend_project/NER/tuned/intention \
  --span_ckpt /cluster/home/gw/Backend_project/NER/tuned/hpoid_span_contrastive_b3mix_multival_unfreeze_lora_hn/hpoid_span_epoch06.pt \
  --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/EVAL/VAL_LLM_full_rerank_B3mix \
  --topk 35 \
  --num_workers 16 \
  --tau_low 0.15 \
  --tau_high 0.45 \
  --hpo_chunk_size 512

Everything else unchanged.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
from tqdm import tqdm

# optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from transformers import AutoTokenizer, AutoModelForCausalLM

# ====== import train-time components ======
from train_hpoid_span_contrastive import (
    HPOConfig,
    HPOOntology,
    TokenCRFWrapper,
    HPOIDSpanPairDataset,
    SpanProj,
    encode_spans,
    encode_hpo_gold_table,
    load_ner_tc_and_tokenizer,
)

logger = logging.getLogger("HPO_MoRE_QWEN_RERANK")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)

SPAN = Tuple[int, int]


# -------------------------------------------------------------------------
# Corpus-set metric helpers
# -------------------------------------------------------------------------
def prf_from_tp_fp_fn(tp: int, fp: int, fn: int) -> Dict[str, float]:
    tp, fp, fn = int(tp), int(fp), int(fn)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def corpus_set_prf(span_rows: List[Dict[str, Any]], pred_field: str) -> Dict[str, Any]:
    gold_set = set()
    pred_set = set()

    for r in span_rows:
        g = r.get("gold_id", None)
        p = r.get(pred_field, None)
        if g:
            gold_set.add(g)
        if p:
            pred_set.add(p)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prf = prf_from_tp_fp_fn(tp, fp, fn)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(prf["precision"]),
        "recall": float(prf["recall"]),
        "f1": float(prf["f1"]),
        "gold_set_size": int(len(gold_set)),
        "pred_set_size": int(len(pred_set)),
    }


# -------------------------------------------------------------------------
# Helper: extract HPO info for prompt (uses llm_def / llm_add_def if present)
# -------------------------------------------------------------------------
def get_hpo_prompt_info(ontology: HPOOntology, hpo_id: str) -> Dict[str, Any]:
    hid = ontology.resolve_id(hpo_id)
    rec = ontology.data.get(hid, {}) or {}

    raw_name = (
        rec.get("Name")
        or rec.get("name")
        or rec.get("label")
        or rec.get("preferred_label")
    )
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

    lines = []
    lines.append(f"[HPO_ID] {hid}")
    lines.append(f"[NAME] {name}")
    if syns:
        lines.append(f"[SYN] {'; '.join(syns)}")
    if orig_def:
        lines.append(f"[DEF] {orig_def}")
    if llm_def and llm_def != orig_def:
        lines.append(f"[LLM_DEF] {llm_def}")
    if llm_add_def:
        lines.append(f"[ADD_DEF] {llm_add_def}")
    combined_def = "\n".join(lines)

    return {
        "hpo_id": hid,
        "hpo_name": name,
        "hpo_def": combined_def,
        "hpo_synonyms": syns,
        "hpo_orig_def": orig_def,
        "hpo_llm_def": llm_def,
        "hpo_add_def": llm_add_def,
    }


# -------------------------------------------------------------------------
# Global HPO table: encode ALL HPO IDs once for all datasets
# -------------------------------------------------------------------------
def build_global_hpo_table(
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer,
    ontology: HPOOntology,
    cfg: HPOConfig,
    device: torch.device,
    chunk_size: int = 512,
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    all_hpo_ids = sorted(list(ontology.data.keys()))
    total = len(all_hpo_ids)
    if total == 0:
        raise RuntimeError("[GlobalHPO] Ontology has no HPO IDs.")

    logger.info(
        f"[GlobalHPO] Building global HPO embedding table for {total} HPO IDs "
        f"(chunk_size={chunk_size})..."
    )

    z_chunks: List[torch.Tensor] = []
    hpo_ids_full: List[str] = []

    for start_idx in tqdm(
        range(0, total, chunk_size),
        desc="[GlobalHPO] Encoding HPO gold table",
        leave=False,
    ):
        chunk_ids = all_hpo_ids[start_idx: start_idx + chunk_size]
        if not chunk_ids:
            continue

        z_hpo_chunk, valid_ids = encode_hpo_gold_table(
            model_tc,
            span_proj,
            tokenizer,
            ontology,
            chunk_ids,
            device=device,
            max_len=cfg.max_len,
        )
        if z_hpo_chunk is None or z_hpo_chunk.numel() == 0:
            continue

        z_chunks.append(z_hpo_chunk)
        hpo_ids_full.extend(valid_ids)

    if not z_chunks:
        raise RuntimeError("[GlobalHPO] Failed to build HPO embedding table (no valid chunks).")

    z_hpo_full = torch.cat(z_chunks, dim=0)
    id2idx_full = {hid: i for i, hid in enumerate(hpo_ids_full)}

    logger.info(
        f"[GlobalHPO] Built global HPO table: "
        f"{z_hpo_full.size(0)} entries, dim={z_hpo_full.size(1)}, device={z_hpo_full.device}"
    )
    return z_hpo_full, hpo_ids_full, id2idx_full


# -------------------------------------------------------------------------
# DualLoRAEnc: returns span_rows (ALL spans) for corpus-set + pipeline assembly
# -------------------------------------------------------------------------
def build_candidates_with_duallora(
    dataset_name: str,
    ds: HPOIDSpanPairDataset,
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer,
    ontology: HPOOntology,
    cfg: HPOConfig,
    device: torch.device,
    z_hpo: torch.Tensor,
    hpo_ids_vec: List[str],
    id2idx: Dict[str, int],
    topk: int = 35,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    BATCH = cfg.batch_size
    dual_top1_hits = 0
    dual_recallK_hits = 0
    total = 0

    samples_for_refine: List[Dict[str, Any]] = []
    span_rows: List[Dict[str, Any]] = []

    logger.info(
        f"[DualLoRAEnc] Building top-{topk} candidates for dataset={dataset_name} "
        f"using global HPO table (size={z_hpo.size(0)})..."
    )

    global_idx = 0

    for i in tqdm(
        range(0, len(ds), BATCH),
        desc=f"[DualLoRAEnc] {dataset_name} - spans",
        leave=False,
    ):
        chunk = [ds[j] for j in range(i, min(i + BATCH, len(ds)))]
        if not chunk:
            continue

        left_texts = [ex["left_text"] for ex in chunk]
        left_spans = [ex["left_span"] for ex in chunk]
        gold_ids = [ex["hpo_id"] for ex in chunk]

        z_left = encode_spans(
            model_tc,
            span_proj,
            tokenizer,
            left_texts,
            left_spans,
            device,
            cfg.max_len,
        )  # [b, D]

        sims = z_left @ z_hpo.t()  # [b, N]
        inner_topk = min(topk, sims.size(1))
        vals, idxs = torch.topk(sims, k=inner_topk, dim=-1)  # [b, topk]

        vals = vals.detach().cpu()
        idxs = idxs.detach().cpu()

        for row, (ex, gold) in enumerate(zip(chunk, gold_ids)):
            total += 1

            gold_idx = id2idx.get(gold, None)
            preds = idxs[row].tolist()

            dual_pred_id: Optional[str] = None
            if preds:
                idx0 = preds[0]
                if 0 <= idx0 < len(hpo_ids_vec):
                    dual_pred_id = hpo_ids_vec[idx0]

            span_rows.append({
                "span_idx": int(global_idx),
                "gold_id": gold,
                "dual_pred_id": dual_pred_id,
            })

            if gold_idx is not None and preds:
                if preds[0] == gold_idx:
                    dual_top1_hits += 1

            gold_in_topk = False
            if gold_idx is not None and gold_idx in preds:
                dual_recallK_hits += 1
                gold_in_topk = True

            if gold_in_topk:
                left_text = ex["left_text"]
                c0, c1 = ex["left_span"]
                c0 = max(0, min(c0, len(left_text)))
                c1 = max(0, min(c1, len(left_text)))
                mention_text = left_text[c0:c1]

                cand_list = []
                for rank_pos in range(inner_topk):
                    idx_hpo = idxs[row, rank_pos].item()
                    score = float(vals[row, rank_pos].item())
                    hid = hpo_ids_vec[idx_hpo]
                    info = get_hpo_prompt_info(ontology, hid)
                    info["score"] = score
                    info["rank"] = int(rank_pos)
                    cand_list.append(info)

                if cand_list:
                    dual_best_id = cand_list[0]["hpo_id"]
                    best_score = cand_list[0]["score"]
                    second_score = cand_list[1]["score"] if len(cand_list) > 1 else best_score
                    margin = best_score - second_score
                else:
                    dual_best_id = None
                    margin = 0.0

                context = left_text[:512] if len(left_text) > 512 else left_text

                samples_for_refine.append({
                    "dataset": dataset_name,
                    "idx": int(global_idx),
                    "context": context,
                    "mention": mention_text,
                    "gold_id": gold,
                    "candidates": cand_list,
                    "dual_best_id": dual_best_id,
                    "dual_margin": float(margin),
                })

            global_idx += 1

    dual_span_top1 = dual_top1_hits / max(1, total)
    dual_recallK = dual_recallK_hits / max(1, total)
    dual_corpus = corpus_set_prf(span_rows, pred_field="dual_pred_id")

    logger.info(
        f"[DualLoRAEnc] dataset={dataset_name} span_top1={dual_span_top1:.4f}, "
        f"recall@{topk}={dual_recallK:.4f}, total_spans={total}, "
        f"refine_samples={len(samples_for_refine)} | "
        f"CORPUS_SET(P/R/F1)={dual_corpus['precision']:.4f}/{dual_corpus['recall']:.4f}/{dual_corpus['f1']:.4f}"
    )

    metrics = {
        "dual_span_top1": float(dual_span_top1),
        "dual_recallK": float(dual_recallK),
        "dual_top1_hits": int(dual_top1_hits),
        "dual_recallK_hits": int(dual_recallK_hits),
        "total_spans": int(total),
        "refine_samples": int(len(samples_for_refine)),

        "dual_corpus_tp": int(dual_corpus["tp"]),
        "dual_corpus_fp": int(dual_corpus["fp"]),
        "dual_corpus_fn": int(dual_corpus["fn"]),
        "dual_corpus_precision": float(dual_corpus["precision"]),
        "dual_corpus_recall": float(dual_corpus["recall"]),
        "dual_corpus_f1": float(dual_corpus["f1"]),
        "dual_gold_set_size": int(dual_corpus["gold_set_size"]),
        "dual_pred_set_size": int(dual_corpus["pred_set_size"]),
    }
    return samples_for_refine, metrics, span_rows


# -------------------------------------------------------------------------
# Local Qwen3-Reranker-4B (pointwise CE) wrapper
# -------------------------------------------------------------------------
def _format_instruction(instruction: str, query: str, doc: str) -> str:
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


class Qwen3Reranker:
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        dtype: str = "float16",
        max_length: int = 8192,
        attn_implementation: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float16)

        kwargs = {"torch_dtype": torch_dtype}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).eval()
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        if self.token_false_id is None or self.token_true_id is None:
            raise RuntimeError("Failed to find token ids for 'yes'/'no'. Check tokenizer vocab.")

    def _process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=True,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        new_input_ids = []
        new_attention = []
        for ids, _attn in zip(inputs["input_ids"], inputs["attention_mask"]):
            ids2 = self.prefix_tokens + ids + self.suffix_tokens
            attn2 = [1] * len(ids2)
            new_input_ids.append(ids2)
            new_attention.append(attn2)

        batch = self.tokenizer.pad(
            {"input_ids": new_input_ids, "attention_mask": new_attention},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        for k in batch:
            batch[k] = batch[k].to(self.model.device)
        return batch

    @torch.no_grad()
    def score_pairs(self, pairs: List[str], batch_size: int = 8) -> List[float]:
        scores: List[float] = []
        n = len(pairs)
        for s in range(0, n, batch_size):
            chunk = pairs[s: s + batch_size]
            inputs = self._process_inputs(chunk)
            logits = self.model(**inputs).logits[:, -1, :]
            yes = logits[:, self.token_true_id]
            no = logits[:, self.token_false_id]
            probs = torch.softmax(torch.stack([no, yes], dim=1), dim=1)
            scores.extend(probs[:, 1].float().cpu().tolist())
        return scores

    def rerank_candidates(
        self,
        instruction: str,
        query: str,
        candidates: List[Dict[str, Any]],
        batch_size: int = 8,
    ) -> Tuple[int, List[float]]:
        pairs = []
        for c in candidates:
            doc = c.get("hpo_def") or ""
            pairs.append(_format_instruction(instruction, query, doc))
        scores = self.score_pairs(pairs, batch_size=batch_size)
        best_i = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1
        return best_i, scores


# -------------------------------------------------------------------------
# Rerank refine with margin-based gating (returns pred_by_idx)
# -------------------------------------------------------------------------
def run_rerank_refine_for_dataset(
    dataset_name: str,
    samples_for_refine: List[Dict[str, Any]],
    reranker: Qwen3Reranker,
    rerank_batch_size: int = 8,
    tau_low: float = 0.05,
    tau_high: float = 0.20,
    rerank_tau: float = 0.05,
) -> Dict[str, Any]:
    if not samples_for_refine:
        logger.warning(f"[RERANK] dataset={dataset_name} has no samples for refine.")
        return {
            "pipeline_top1": 0.0,
            "pipeline_top1_hits": 0,
            "n_samples": 0,
            "refine_calls": 0,
            "rerank_top1_hits": 0,
            "rerank_top1_conditional": 0.0,
            "pred_by_idx": {},
        }

    instruction = (
        "Given a mention and its clinical context, judge whether a candidate HPO term "
        "is the best match for the mention. Answer yes if it is the best match, otherwise no."
    )

    logger.info(
        f"[RERANK] Starting margin-gated refine on dataset={dataset_name} with {len(samples_for_refine)} samples..."
    )

    n = len(samples_for_refine)
    pipeline_hits = 0
    refine_calls = 0
    rerank_hits = 0
    pred_by_idx: Dict[int, str] = {}

    for sample in tqdm(samples_for_refine, desc=f"[RERANK] {dataset_name} - gated refine", leave=False):
        sid = int(sample["idx"])
        context = sample["context"]
        mention = sample["mention"]
        gold_id = sample["gold_id"]
        candidates = sample["candidates"]
        dual_best_id = sample.get("dual_best_id", None)
        margin = float(sample.get("dual_margin", 0.0))

        if dual_best_id is None or not candidates:
            continue

        query = (
            f"Mention: {mention}\n"
            f"Context:\n{context}\n"
            f"Task: Select the single best matching HPO term for the mention in this context."
        )

        if margin >= tau_high:
            pred_final = dual_best_id
            pred_by_idx[sid] = pred_final
            if pred_final == gold_id:
                pipeline_hits += 1
            continue

        refine_calls += 1
        best_i, scores = reranker.rerank_candidates(
            instruction=instruction,
            query=query,
            candidates=candidates,
            batch_size=rerank_batch_size,
        )

        if best_i < 0 or best_i >= len(candidates) or not scores:
            pred_final = dual_best_id
            pred_by_idx[sid] = pred_final
            if pred_final == gold_id:
                pipeline_hits += 1
            continue

        rerank_hid = candidates[best_i]["hpo_id"]
        if rerank_hid == gold_id:
            rerank_hits += 1

        s_sorted = sorted(scores, reverse=True)
        top1 = s_sorted[0]
        top2 = s_sorted[1] if len(s_sorted) > 1 else s_sorted[0]
        rerank_margin = float(top1 - top2)

        if margin <= tau_low:
            pred_final = rerank_hid
        else:
            pred_final = rerank_hid if rerank_margin >= rerank_tau else dual_best_id

        pred_by_idx[sid] = pred_final
        if pred_final == gold_id:
            pipeline_hits += 1

    pipeline_top1 = pipeline_hits / max(1, n)
    rerank_cond = (rerank_hits / refine_calls) if refine_calls > 0 else 0.0

    logger.info(
        f"[RERANK] dataset={dataset_name} pipeline_top1={pipeline_top1:.4f} "
        f"(pipeline_hits={pipeline_hits}/{n}), refine_calls={refine_calls}, "
        f"rerank_cond_top1={rerank_cond:.4f} (rerank_hits={rerank_hits})"
    )

    return {
        "pipeline_top1": float(pipeline_top1),
        "pipeline_top1_hits": int(pipeline_hits),
        "n_samples": int(n),
        "refine_calls": int(refine_calls),
        "rerank_top1_hits": int(rerank_hits),
        "rerank_top1_conditional": float(rerank_cond),
        "pred_by_idx": pred_by_idx,
    }


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------
def plot_comparisons(
    results_summary: Dict[str, Dict[str, Any]],
    out_dir: str,
    topk: int,
) -> Dict[str, str]:
    if not HAS_MPL:
        logger.warning("[PLOT] matplotlib not available, skip plotting.")
        return {"top1": "", "recall_rerank": ""}

    dataset_names = [k for k in results_summary.keys() if not k.startswith("_")]
    if not dataset_names:
        return {"top1": "", "recall_rerank": ""}

    dataset_names = sorted(dataset_names)
    dual_top1_vals = []
    pipe_top1_vals = []
    recall_vals = []
    rerank_cond_vals = []

    for ds in dataset_names:
        m = results_summary[ds]
        dual_top1_vals.append(float(m.get("dual_span_top1", 0.0)))
        pipe_top1_vals.append(float(m.get("pipeline_span_top1", 0.0)))
        recall_vals.append(float(m.get("dual_recallK", 0.0)))
        rerank_cond_vals.append(float(m.get("rerank_top1_conditional", 0.0)))

    x = list(range(len(dataset_names)))
    width = 0.35
    fig_w = max(6.0, 1.5 * len(dataset_names))

    plt.style.use("ggplot")

    plt.figure(figsize=(fig_w, 4.5))
    plt.bar([xi - width / 2 for xi in x], dual_top1_vals, width=width, label="DualLoRAEnc Span Top-1")
    plt.bar([xi + width / 2 for xi in x], pipe_top1_vals, width=width, label="Pipeline Span Top-1 (Dual+Rerank)")
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Span Top-1")
    plt.title("Span Top-1: DualLoRAEnc vs Pipeline (Qwen3-Reranker, margin-gated)")
    plt.legend()
    plt.tight_layout()
    top1_path = os.path.join(out_dir, "top1_comparison.png")
    plt.savefig(top1_path, dpi=200)
    plt.close()

    plt.figure(figsize=(fig_w, 4.5))
    plt.bar([xi - width / 2 for xi in x], recall_vals, width=width, label=f"DualLoRAEnc Recall@{topk}")
    plt.bar([xi + width / 2 for xi in x], rerank_cond_vals, width=width, label="Reranker Conditional Top-1 (called)")
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Ratio")
    plt.title(f"Recall@{topk} vs Reranker Conditional Top-1")
    plt.legend()
    plt.tight_layout()
    recall_path = os.path.join(out_dir, "recall_vs_rerank.png")
    plt.savefig(recall_path, dpi=200)
    plt.close()

    logger.info(f"[PLOT] Saved plots to {top1_path} and {recall_path}")
    return {"top1": top1_path, "recall_rerank": recall_path}


# -------------------------------------------------------------------------
# CLI & main
# -------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="DualLoRAEnc eval + LOCAL Qwen3-Reranker refine for HPO-ID with margin-based gating (global HPO table)."
    )
    ap.add_argument(
        "--eval_roots",
        type=str,
        nargs="+",
        required=True,
        help="Eval roots (GeneReviews / GSC+ / ID-68), each with ann/ and corpus/.",
    )

    # (CHANGED) default val_root path (so you can pass exactly the line you asked)
    ap.add_argument(
        "--val_root",
        type=str,
        default="/cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/val",
        help="Optional extra validation root; evaluated as an additional dataset (default: phenobert/data/val).",
    )

    ap.add_argument("--hpo_json", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, required=True)
    ap.add_argument("--init_encoder_from", type=str, default=None)
    ap.add_argument("--span_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=35)

    ap.add_argument("--tau_low", type=float, default=0.05)
    ap.add_argument("--tau_high", type=float, default=0.20)
    ap.add_argument("--hpo_chunk_size", type=int, default=512)

    ap.add_argument("--reranker_model_dir", type=str, required=True)
    ap.add_argument("--rerank_batch_size", type=int, default=8)
    ap.add_argument("--rerank_tau", type=float, default=0.05)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--flash_attn2", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = HPOConfig(
        backbone=args.backbone,
        init_encoder_from=args.init_encoder_from,
        model_dir=args.model_dir,
        hpo_json=args.hpo_json,
        max_len=args.max_len,
        batch_size=args.batch_size,
        stride=0,
        hpo_topk=args.topk,
    )

    ontology = HPOOntology(args.hpo_json)
    logger.info(f"Loaded HPO ontology with {len(ontology.data)} nodes from {args.hpo_json}.")

    tokenizer, model_tc, meta = load_ner_tc_and_tokenizer(
        args.backbone,
        args.init_encoder_from,
        args.model_dir,
        cfg,
    )
    model_tc.to(device)
    model_tc.eval()
    hidden_size = model_tc.base.config.hidden_size

    if not os.path.isfile(args.span_ckpt):
        raise FileNotFoundError(f"Span checkpoint not found: {args.span_ckpt}")
    ckpt = torch.load(args.span_ckpt, map_location="cpu")
    span_dim = ckpt.get("cfg", {}).get("hpoid_dim", 256)
    span_proj = SpanProj(in_dim=hidden_size, out_dim=span_dim, dropout=0.0).to(device)
    span_proj.load_state_dict(ckpt["span_proj_state"])
    span_proj.eval()
    logger.info(f"Loaded span projection head from {args.span_ckpt} (epoch={ckpt.get('epoch','?')}).")

    logger.info(f"Loading local reranker from: {args.reranker_model_dir}")
    reranker = Qwen3Reranker(
        model_dir=args.reranker_model_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=args.dtype,
        max_length=8192,
        attn_implementation="flash_attention_2" if args.flash_attn2 else None,
    )

    cfg_out = {"cli": vars(args), "meta": meta, "reranker": {"model_dir": args.reranker_model_dir}}
    cfg_path = os.path.join(args.out_dir, "hpo_more_qwen_rerank_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved config to {cfg_path}")

    z_hpo_global, hpo_ids_global, id2idx_global = build_global_hpo_table(
        model_tc=model_tc,
        span_proj=span_proj,
        tokenizer=tokenizer,
        ontology=ontology,
        cfg=cfg,
        device=device,
        chunk_size=args.hpo_chunk_size,
    )

    results_summary: Dict[str, Dict[str, Any]] = {}

    global_span_rows: List[Dict[str, Any]] = []
    global_total_spans = 0
    global_dual_top1_hits = 0
    global_dual_recallK_hits = 0
    global_pipeline_top1_hits_over_all_spans = 0

    global_refine_samples = 0
    global_refine_calls = 0
    global_rerank_top1_hits = 0

    # (CHANGED) always append val_root if provided (default is set to your path)
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

        samples_for_refine, dual_metrics, span_rows = build_candidates_with_duallora(
            dataset_name=dataset_name,
            ds=ds,
            model_tc=model_tc,
            span_proj=span_proj,
            tokenizer=tokenizer,
            ontology=ontology,
            cfg=cfg,
            device=device,
            z_hpo=z_hpo_global,
            hpo_ids_vec=hpo_ids_global,
            id2idx=id2idx_global,
            topk=args.topk,
        )

        rerank_metrics = run_rerank_refine_for_dataset(
            dataset_name=dataset_name,
            samples_for_refine=samples_for_refine,
            reranker=reranker,
            rerank_batch_size=args.rerank_batch_size,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            rerank_tau=args.rerank_tau,
        )

        pred_by_idx = rerank_metrics.get("pred_by_idx", {}) or {}

        pipeline_span_top1_hits_over_all = 0
        for r in span_rows:
            sid = int(r["span_idx"])
            pipeline_pred = pred_by_idx.get(sid, r.get("dual_pred_id", None))
            r["pipeline_pred_id"] = pipeline_pred
            if pipeline_pred and pipeline_pred == r["gold_id"]:
                pipeline_span_top1_hits_over_all += 1

        total_spans = int(dual_metrics["total_spans"])
        pipeline_span_top1 = pipeline_span_top1_hits_over_all / max(1, total_spans)
        pipe_corpus = corpus_set_prf(span_rows, pred_field="pipeline_pred_id")

        results_summary[dataset_name] = {
            "total_spans": total_spans,
            "dual_span_top1": float(dual_metrics["dual_span_top1"]),
            "dual_recallK": float(dual_metrics["dual_recallK"]),
            "pipeline_span_top1": float(pipeline_span_top1),

            "dual_corpus_tp": int(dual_metrics["dual_corpus_tp"]),
            "dual_corpus_fp": int(dual_metrics["dual_corpus_fp"]),
            "dual_corpus_fn": int(dual_metrics["dual_corpus_fn"]),
            "dual_corpus_precision": float(dual_metrics["dual_corpus_precision"]),
            "dual_corpus_recall": float(dual_metrics["dual_corpus_recall"]),
            "dual_corpus_f1": float(dual_metrics["dual_corpus_f1"]),

            "pipeline_corpus_tp": int(pipe_corpus["tp"]),
            "pipeline_corpus_fp": int(pipe_corpus["fp"]),
            "pipeline_corpus_fn": int(pipe_corpus["fn"]),
            "pipeline_corpus_precision": float(pipe_corpus["precision"]),
            "pipeline_corpus_recall": float(pipe_corpus["recall"]),
            "pipeline_corpus_f1": float(pipe_corpus["f1"]),

            "refine_samples": int(dual_metrics["refine_samples"]),
            "refine_calls": int(rerank_metrics["refine_calls"]),
            "rerank_top1_conditional": float(rerank_metrics["rerank_top1_conditional"]),
        }

        global_total_spans += total_spans
        global_dual_top1_hits += int(dual_metrics["dual_top1_hits"])
        global_dual_recallK_hits += int(dual_metrics["dual_recallK_hits"])
        global_pipeline_top1_hits_over_all_spans += int(pipeline_span_top1_hits_over_all)

        global_refine_samples += int(dual_metrics["refine_samples"])
        global_refine_calls += int(rerank_metrics["refine_calls"])
        global_rerank_top1_hits += int(rerank_metrics["rerank_top1_hits"])

        global_span_rows.extend(span_rows)

    global_dual_span_top1 = global_dual_top1_hits / max(1, global_total_spans)
    global_dual_recallK = global_dual_recallK_hits / max(1, global_total_spans)
    global_pipeline_span_top1 = global_pipeline_top1_hits_over_all_spans / max(1, global_total_spans)
    global_rerank_cond = (global_rerank_top1_hits / global_refine_calls) if global_refine_calls > 0 else 0.0

    global_dual_corpus = corpus_set_prf(global_span_rows, pred_field="dual_pred_id")
    global_pipe_corpus = corpus_set_prf(global_span_rows, pred_field="pipeline_pred_id")

    results_summary["_GLOBAL"] = {
        "total_spans": int(global_total_spans),
        "dual_span_top1": float(global_dual_span_top1),
        "dual_recallK": float(global_dual_recallK),
        "pipeline_span_top1": float(global_pipeline_span_top1),
        "rerank_top1_conditional": float(global_rerank_cond),

        "dual_corpus_tp": int(global_dual_corpus["tp"]),
        "dual_corpus_fp": int(global_dual_corpus["fp"]),
        "dual_corpus_fn": int(global_dual_corpus["fn"]),
        "dual_corpus_precision": float(global_dual_corpus["precision"]),
        "dual_corpus_recall": float(global_dual_corpus["recall"]),
        "dual_corpus_f1": float(global_dual_corpus["f1"]),

        "pipeline_corpus_tp": int(global_pipe_corpus["tp"]),
        "pipeline_corpus_fp": int(global_pipe_corpus["fp"]),
        "pipeline_corpus_fn": int(global_pipe_corpus["fn"]),
        "pipeline_corpus_precision": float(global_pipe_corpus["precision"]),
        "pipeline_corpus_recall": float(global_pipe_corpus["recall"]),
        "pipeline_corpus_f1": float(global_pipe_corpus["f1"]),

        "refine_samples": int(global_refine_samples),
        "refine_calls": int(global_refine_calls),
        "rerank_top1_hits": int(global_rerank_top1_hits),
    }

    summary_json_path = os.path.join(args.out_dir, "hpo_more_qwen_rerank_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON summary to {summary_json_path}")

    plot_paths = plot_comparisons(results_summary, args.out_dir, topk=args.topk)

    md_path = os.path.join(args.out_dir, "hpo_more_qwen_rerank_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPO DualLoRAEnc + Qwen3-Reranker Evaluation (with val_root)\n\n")
        f.write("## Command\n\n```bash\n")
        f.write("python HPO_MoRE_QWEN_RERANK.py \\\n")
        for k, v in vars(args).items():
            if isinstance(v, list):
                for item in v:
                    f.write(f"  --{k} {item} \\\n")
            elif isinstance(v, bool):
                if v:
                    f.write(f"  --{k} \\\n")
            else:
                f.write(f"  --{k} {v} \\\n")
        f.write("```\n\n")

        g = results_summary["_GLOBAL"]
        f.write("## Global Metrics\n\n")
        f.write(f"- Total spans: **{g['total_spans']}**\n")
        f.write(f"- Dual span top-1: **{g['dual_span_top1']:.4f}**\n")
        f.write(f"- Dual recall@{args.topk}: **{g['dual_recallK']:.4f}**\n")
        f.write(f"- Pipeline span top-1: **{g['pipeline_span_top1']:.4f}**\n")
        f.write(f"- Reranker conditional top-1 (called): **{g['rerank_top1_conditional']:.4f}**\n\n")

        f.write("### Corpus-set PRF\n\n")
        f.write(
            f"- Dual corpus P/R/F1: **{g['dual_corpus_precision']:.4f}/{g['dual_corpus_recall']:.4f}/{g['dual_corpus_f1']:.4f}** "
            f"(TP/FP/FN={g['dual_corpus_tp']}/{g['dual_corpus_fp']}/{g['dual_corpus_fn']})\n"
        )
        f.write(
            f"- Pipeline corpus P/R/F1: **{g['pipeline_corpus_precision']:.4f}/{g['pipeline_corpus_recall']:.4f}/{g['pipeline_corpus_f1']:.4f}** "
            f"(TP/FP/FN={g['pipeline_corpus_tp']}/{g['pipeline_corpus_fp']}/{g['pipeline_corpus_fn']})\n\n"
        )

        f.write("## Per-dataset Metrics\n\n")
        f.write(
            f"| Dataset | Spans | Dual span top1 | Dual recall@{args.topk} | Pipe span top1 | Rerank calls | Rerank cond | Dual corpus F1 | Pipe corpus F1 |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for ds_name, m in results_summary.items():
            if ds_name.startswith("_"):
                continue
            f.write(
                f"| {ds_name} | {m['total_spans']} | {m['dual_span_top1']:.4f} | {m['dual_recallK']:.4f} | "
                f"{m['pipeline_span_top1']:.4f} | {m['refine_calls']} | {m['rerank_top1_conditional']:.4f} | "
                f"{m['dual_corpus_f1']:.4f} | {m['pipeline_corpus_f1']:.4f} |\n"
            )

        f.write("\n## Plots\n\n")
        if plot_paths.get("top1"):
            f.write(f"![Span Top-1 Comparison]({os.path.basename(plot_paths['top1'])})\n\n")
        if plot_paths.get("recall_rerank"):
            f.write(f"![Recall vs Reranker Conditional]({os.path.basename(plot_paths['recall_rerank'])})\n\n")

    logger.info(f"Markdown summary saved to: {md_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
