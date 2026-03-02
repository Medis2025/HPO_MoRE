#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPO_MoRE.py

DualLoRAEnc + LLM refine eval for HPO-ID with margin-based gating,
using a GLOBAL HPO table and enriched HPO JSON (with llm_def / llm_add_def).

(REVISION ONLY): report **CORPUS-SET** TP/FP/FN + Precision/Recall/F1 for:
  - DualLoRAEnc top-1 predictions (per corpus + global)
  - Full Pipeline top-1 predictions (per corpus + global)

Corpus-set definition (within each dataset/corpus):
  For a corpus C:
    gold_set(C) = set(all gold HPO IDs across all spans in the corpus)
    pred_set(C) = set(all predicted top-1 HPO IDs across all spans in the corpus)

  TP(C) = |pred_set ∩ gold_set|
  FP(C) = |pred_set - gold_set|
  FN(C) = |gold_set - pred_set|

  Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R)

We still keep span-level top1 and recall@K because they are useful for retrieval.

Example:
python /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/HPO_More_full.py \
  --eval_roots \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GeneReviews \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GSC+ \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/ID-68 \
  --val_root /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/val \
  --hpo_json /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json \
  --model_dir /cluster/home/gw/Backend_project/NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --init_encoder_from /cluster/home/gw/Backend_project/NER/tuned/intention \
  --span_ckpt /cluster/home/gw/Backend_project/NER/tuned/duallora_span_llm/hpoid_span_best.pt \
  --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/VAL_LLM_full \
  --prompt_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/prompts/select_hpo.txt \
  --api_key_env DEEPSEEK_API_KEY \
  --topk 35 \
  --num_workers 16 \
  --tau_low 0.05 \
  --tau_high 0.20 \
  --hpo_chunk_size 512
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm

# optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

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

# ====== import LLM client & refiner ======
from hpo_llm_refiner import LLMAPIClient, HPOCandidateRefiner

logger = logging.getLogger("HPO_Revised_LLM")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)

SPAN = Tuple[int, int]


# -------------------------------------------------------------------------
# (NEW) Corpus-set metric helpers (span_top1/recall@K + corpus_set_F1 only)
# -------------------------------------------------------------------------
def prf_from_tp_fp_fn(tp: int, fp: int, fn: int) -> Dict[str, float]:
    tp, fp, fn = int(tp), int(fp), int(fn)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def corpus_set_prf(span_rows: List[Dict[str, Any]], pred_field: str) -> Dict[str, Any]:
    """
    span_rows items must contain:
      - gold_id
      - pred_field (predicted hpo id for that span)
    Compute corpus-level set TP/FP/FN and PRF.
    """
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
# Helper: extract HPO info for prompt (unchanged)
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
# Global HPO table (unchanged)
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
# DualLoRAEnc: now also returns span_rows for corpus-set metrics
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
    topk: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns:
      samples_for_llm: list of samples where gold ∈ topK
      metrics: span-level dual metrics (top1, recall@K, etc.) + corpus-set F1
      span_rows: list of rows for corpus-set metrics:
        {
          "span_idx": int,
          "gold_id": str,
          "dual_pred_id": str|None,
        }
    """
    BATCH = cfg.batch_size
    dual_top1_hits = 0
    dual_recallK_hits = 0
    total = 0

    samples_for_llm: List[Dict[str, Any]] = []
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

            left_text = ex["left_text"]
            c0, c1 = ex["left_span"]
            c0 = max(0, min(c0, len(left_text)))
            c1 = max(0, min(c1, len(left_text)))
            mention_text = left_text[c0:c1]

            gold_idx = id2idx.get(gold, None)
            preds = idxs[row].tolist()

            dual_pred_id: Optional[str] = None
            if preds:
                idx0 = preds[0]
                if 0 <= idx0 < len(hpo_ids_vec):
                    dual_pred_id = hpo_ids_vec[idx0]

            # record corpus-set span row
            span_rows.append({
                "span_idx": int(global_idx),
                "gold_id": gold,
                "dual_pred_id": dual_pred_id,
            })

            # Dual top-1 hit
            if gold_idx is not None and preds:
                if preds[0] == gold_idx:
                    dual_top1_hits += 1

            # recall@K
            gold_in_topk = False
            if gold_idx is not None and gold_idx in preds:
                dual_recallK_hits += 1
                gold_in_topk = True

            # Only build LLM samples if gold ∈ topK (unchanged behavior)
            if gold_in_topk:
                cand_list = []
                for rank_pos in range(inner_topk):
                    idx_hpo = idxs[row, rank_pos].item()
                    score = float(vals[row, rank_pos].item())
                    hid = hpo_ids_vec[idx_hpo]
                    info = get_hpo_prompt_info(ontology, hid)
                    info["score"] = score
                    cand_list.append(info)

                if cand_list:
                    dual_best_id = cand_list[0]["hpo_id"]
                    best_score = cand_list[0]["score"]
                    if len(cand_list) > 1:
                        second_score = cand_list[1]["score"]
                    else:
                        second_score = best_score
                    margin = best_score - second_score
                else:
                    dual_best_id = None
                    margin = 0.0

                context = left_text[:512] if len(left_text) > 512 else left_text

                samples_for_llm.append({
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

    dual_top1 = dual_top1_hits / max(1, total)
    dual_recallK = dual_recallK_hits / max(1, total)

    # NEW: corpus-set PRF for Dual predictions
    dual_corpus = corpus_set_prf(span_rows, pred_field="dual_pred_id")

    logger.info(
        f"[DualLoRAEnc] dataset={dataset_name} span_top1={dual_top1:.4f}, "
        f"recall@{topk}={dual_recallK:.4f}, total_spans={total}, "
        f"LLM_samples={len(samples_for_llm)} | "
        f"CORPUS_SET(P/R/F1)={dual_corpus['precision']:.4f}/{dual_corpus['recall']:.4f}/{dual_corpus['f1']:.4f} "
        f"(TP/FP/FN={dual_corpus['tp']}/{dual_corpus['fp']}/{dual_corpus['fn']}, "
        f"|G|={dual_corpus['gold_set_size']}, |P|={dual_corpus['pred_set_size']})"
    )

    metrics = {
        # span-level
        "dual_span_top1": float(dual_top1),
        "dual_recallK": float(dual_recallK),
        "dual_top1_hits": int(dual_top1_hits),
        "dual_recallK_hits": int(dual_recallK_hits),
        "total_spans": int(total),
        "llm_samples": int(len(samples_for_llm)),
        # corpus-set
        "dual_corpus_precision": float(dual_corpus["precision"]),
        "dual_corpus_recall": float(dual_corpus["recall"]),
        "dual_corpus_f1": float(dual_corpus["f1"]),
        "dual_corpus_tp": int(dual_corpus["tp"]),
        "dual_corpus_fp": int(dual_corpus["fp"]),
        "dual_corpus_fn": int(dual_corpus["fn"]),
        "gold_set_size": int(dual_corpus["gold_set_size"]),
        "pred_set_size": int(dual_corpus["pred_set_size"]),
    }
    return samples_for_llm, metrics, span_rows


# -------------------------------------------------------------------------
# LLM refine: unchanged algorithm, but now also returns per-span pipeline predictions
# -------------------------------------------------------------------------
def run_llm_refine_for_dataset(
    dataset_name: str,
    samples_for_llm: List[Dict[str, Any]],
    refiner: HPOCandidateRefiner,
    num_workers: int = 16,
    tau_low: float = 0.05,
    tau_high: float = 0.20,
) -> Dict[str, Any]:
    """
    IMPORTANT behavior unchanged:
      - Only runs on samples where gold ∈ topK.
      - pipeline_hits is counted over those samples.
    NEW: returns per-sample final pred_id for these samples, keyed by sample['idx'].
    """
    if not samples_for_llm:
        logger.warning(f"[LLM] dataset={dataset_name} has no samples for LLM refine.")
        return {
            "pipeline_top1": 0.0,
            "pipeline_top1_hits": 0,
            "n_samples": 0,
            "llm_calls": 0,
            "llm_top1_hits": 0,
            "llm_top1_conditional": 0.0,
            "pred_by_idx": {},  # idx -> final_pred_id (for corpus-set assembly)
        }

    logger.info(
        f"[LLM] Starting margin-gated refine on dataset={dataset_name} with {len(samples_for_llm)} samples..."
    )

    n = len(samples_for_llm)
    pred_by_idx: Dict[int, str] = {}

    def _run_one(sample: Dict[str, Any]):
        """
        Returns:
          (idx, pipeline_hit: bool, llm_called: bool, llm_hit: bool, pred_final_id: str)
        """
        sid = int(sample["idx"])
        gold_id = sample["gold_id"]
        candidates = sample["candidates"]
        dual_best_id = sample.get("dual_best_id", None)
        margin = float(sample.get("dual_margin", 0.0))

        if dual_best_id is None:
            return sid, False, False, False, ""

        # Easy: trust Dual, no call
        if margin >= tau_high:
            pred_final = dual_best_id
            pipeline_hit = (pred_final == gold_id)
            return sid, pipeline_hit, False, False, pred_final

        llm_called = True
        pred_final = dual_best_id  # fallback
        llm_hit = False

        try:
            indices = refiner.refine(sample["context"], sample["mention"], candidates)
        except Exception as e:
            logger.warning(f"[LLM] Error in refine: {e}")
            pipeline_hit = (pred_final == gold_id)
            return sid, pipeline_hit, llm_called, llm_hit, pred_final

        if not indices:
            pipeline_hit = (pred_final == gold_id)
            return sid, pipeline_hit, llm_called, llm_hit, pred_final

        pred_idx = indices[0]
        if pred_idx < 0 or pred_idx >= len(candidates):
            pipeline_hit = (pred_final == gold_id)
            return sid, pipeline_hit, llm_called, llm_hit, pred_final

        llm_hid = candidates[pred_idx]["hpo_id"]
        llm_hit = (llm_hid == gold_id)

        if margin <= tau_low:
            pred_final = llm_hid  # hard: trust LLM
            pipeline_hit = (pred_final == gold_id)
            return sid, pipeline_hit, llm_called, llm_hit, pred_final

        # medium: override only if correct
        pred_final = llm_hid if llm_hit else dual_best_id
        pipeline_hit = (pred_final == gold_id)
        return sid, pipeline_hit, llm_called, llm_hit, pred_final

    pipeline_hits = 0
    llm_calls = 0
    llm_hits = 0

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [pool.submit(_run_one, s) for s in samples_for_llm]
        for fut in tqdm(
            as_completed(futures),
            total=n,
            desc=f"[LLM] {dataset_name} - gated refine",
            leave=False,
        ):
            try:
                sid, pipeline_hit, llm_called, llm_hit, pred_final = fut.result()
            except Exception as e:
                logger.warning(f"[LLM] Future error: {e}")
                continue

            if pred_final:
                pred_by_idx[sid] = pred_final

            if pipeline_hit:
                pipeline_hits += 1
            if llm_called:
                llm_calls += 1
                if llm_hit:
                    llm_hits += 1

    pipeline_top1 = pipeline_hits / max(1, n)
    llm_cond = (llm_hits / llm_calls) if llm_calls > 0 else 0.0

    logger.info(
        f"[LLM] dataset={dataset_name} pipeline_top1={pipeline_top1:.4f} "
        f"(pipeline_hits={pipeline_hits}/{n}), "
        f"llm_calls={llm_calls}, llm_cond_top1={llm_cond:.4f} (llm_hits={llm_hits})"
    )

    return {
        "pipeline_top1": float(pipeline_top1),
        "pipeline_top1_hits": int(pipeline_hits),
        "n_samples": int(n),
        "llm_calls": int(llm_calls),
        "llm_top1_hits": int(llm_hits),
        "llm_top1_conditional": float(llm_cond),
        "pred_by_idx": pred_by_idx,
    }


# -------------------------------------------------------------------------
# Plotting helpers (unchanged)
# -------------------------------------------------------------------------
def plot_comparisons(
    results_summary: Dict[str, Dict[str, Any]],
    out_dir: str,
    topk: int,
) -> Dict[str, str]:
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
        dual_top1_vals.append(float(m.get("dual_span_top1", 0.0)))
        pipe_top1_vals.append(float(m.get("pipeline_span_top1", 0.0)))
        recall_vals.append(float(m.get("dual_recallK", 0.0)))
        llm_cond_vals.append(float(m.get("llm_top1_conditional", 0.0)))

    x = list(range(len(dataset_names)))
    width = 0.35

    plt.style.use("ggplot")
    fig_w = max(6.0, 1.5 * len(dataset_names))

    plt.figure(figsize=(fig_w, 4.5))
    plt.bar([xi - width / 2 for xi in x], dual_top1_vals, width=width, label="DualLoRAEnc Span Top-1")
    plt.bar([xi + width / 2 for xi in x], pipe_top1_vals, width=width, label="Pipeline Span Top-1")
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Span Top-1")
    plt.title("Span Top-1: DualLoRAEnc vs Pipeline")
    plt.legend()
    plt.tight_layout()

    top1_path = os.path.join(out_dir, "top1_comparison.png")
    plt.savefig(top1_path, dpi=200)
    plt.close()

    plt.figure(figsize=(fig_w, 4.5))
    plt.bar([xi - width / 2 for xi in x], recall_vals, width=width, label=f"DualLoRAEnc Recall@{topk}")
    plt.bar([xi + width / 2 for xi in x], llm_cond_vals, width=width, label="LLM Conditional Top-1 (called)")
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
# CLI & main
# -------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="DualLoRAEnc eval + LLM refine for HPO-ID (topK candidates) with margin-based gating (global HPO table)."
    )
    ap.add_argument("--eval_roots", type=str, nargs="+", required=True,
                    help="Eval roots (GeneReviews / GSC+ / ID-68), each with ann/ and corpus/.")
    ap.add_argument("--val_root", type=str, default=None,
                    help="Optional extra validation root; if set, it is evaluated as an additional dataset.")
    ap.add_argument("--hpo_json", type=str, required=True,
                    help="Path to hpo.json (enriched_with_llm.json).")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="NER LoRA model dir (TokenCRFWrapper / PeftModel).")
    ap.add_argument("--backbone", type=str, required=True,
                    help="HF backbone path (e.g., BioLinkBERT-base).")
    ap.add_argument("--init_encoder_from", type=str, default=None,
                    help="Optional encoder init checkpoint.")
    ap.add_argument("--span_ckpt", type=str, required=True,
                    help="Path to span projection checkpoint.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for eval json, markdown and plots.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=15, help="topK candidates for LLM refine.")

    ap.add_argument("--prompt_path", type=str, required=True,
                    help="Path to prompt txt file (used by LLMAPIClient).")
    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY",
                    help="Env var name for LLM API key.")
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com",
                    help="Base URL of LLM API.")
    ap.add_argument("--model", type=str, default="deepseek-chat",
                    help="LLM model name.")
    ap.add_argument("--num_workers", type=int, default=16,
                    help="Max threads for LLM refine.")

    ap.add_argument("--tau_low", type=float, default=0.05,
                    help="Margin threshold for HARD cases (margin <= tau_low -> trust LLM).")
    ap.add_argument("--tau_high", type=float, default=0.20,
                    help="Margin threshold for EASY cases (margin >= tau_high -> trust Dual).")

    ap.add_argument("--hpo_chunk_size", type=int, default=512,
                    help="Chunk size when encoding the global HPO table (to avoid OOM).")
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

    api_key = os.environ.get(args.api_key_env, None)
    if not api_key:
        raise RuntimeError(
            f"Env var {args.api_key_env} is not set. Please: export {args.api_key_env}='sk-xxxx'"
        )

    llm_client = LLMAPIClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=60.0,
        prompt_path=args.prompt_path,
    )
    refiner = HPOCandidateRefiner(llm_client, max_candidates=args.topk)

    cfg_out = {"cli": vars(args), "meta": meta}
    cfg_path = os.path.join(args.out_dir, "hpo_revise_llm_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved config to {cfg_path}")

    # Build global HPO table
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

    # ---------- Global aggregation storage for CORPUS metrics ----------
    global_span_rows: List[Dict[str, Any]] = []  # will be filled with dual_pred_id + pipeline_pred_id
    global_total_spans = 0
    global_dual_top1_hits = 0
    global_dual_recallK_hits = 0
    global_pipeline_top1_hits_over_all_spans = 0

    global_llm_samples = 0
    global_llm_calls = 0
    global_llm_top1_hits = 0

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

        # Dual: candidates + span_rows for corpus-set
        samples_for_llm, dual_metrics, span_rows = build_candidates_with_duallora(
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

        # LLM gating on subset (gold ∈ topK)
        llm_metrics = run_llm_refine_for_dataset(
            dataset_name,
            samples_for_llm,
            refiner,
            num_workers=args.num_workers,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
        )

        # ---------- Build PIPELINE prediction for ALL spans (NO fallback) ----------
        # pipeline_pred_id = refined_pred (if available for that span_idx) else dual_pred_id
        pred_by_idx = llm_metrics.get("pred_by_idx", {}) or {}

        pipeline_span_top1_hits_over_all = 0
        for r in span_rows:
            sid = int(r["span_idx"])
            pipeline_pred = pred_by_idx.get(sid, r["dual_pred_id"])
            r["pipeline_pred_id"] = pipeline_pred
            if pipeline_pred and pipeline_pred == r["gold_id"]:
                pipeline_span_top1_hits_over_all += 1

        total_spans = int(dual_metrics["total_spans"])
        pipeline_span_top1 = pipeline_span_top1_hits_over_all / max(1, total_spans)

        # CORPUS-set pipeline metrics
        pipe_corpus = corpus_set_prf(span_rows, pred_field="pipeline_pred_id")

        # store per-dataset summary
        results_summary[dataset_name] = {
            # span-level
            "total_spans": total_spans,
            "dual_span_top1": float(dual_metrics["dual_span_top1"]),
            "dual_recallK": float(dual_metrics["dual_recallK"]),
            "pipeline_span_top1": float(pipeline_span_top1),

            # corpus-set
            "dual_corpus_tp": int(dual_metrics["dual_corpus_tp"]),
            "dual_corpus_fp": int(dual_metrics["dual_corpus_fp"]),
            "dual_corpus_fn": int(dual_metrics["dual_corpus_fn"]),
            "dual_corpus_precision": float(dual_metrics["dual_corpus_precision"]),
            "dual_corpus_recall": float(dual_metrics["dual_corpus_recall"]),
            "dual_corpus_f1": float(dual_metrics["dual_corpus_f1"]),
            "dual_gold_set_size": int(dual_metrics["gold_set_size"]),
            "dual_pred_set_size": int(dual_metrics["pred_set_size"]),

            "pipeline_corpus_tp": int(pipe_corpus["tp"]),
            "pipeline_corpus_fp": int(pipe_corpus["fp"]),
            "pipeline_corpus_fn": int(pipe_corpus["fn"]),
            "pipeline_corpus_precision": float(pipe_corpus["precision"]),
            "pipeline_corpus_recall": float(pipe_corpus["recall"]),
            "pipeline_corpus_f1": float(pipe_corpus["f1"]),
            "pipeline_gold_set_size": int(pipe_corpus["gold_set_size"]),
            "pipeline_pred_set_size": int(pipe_corpus["pred_set_size"]),

            # LLM side
            "llm_samples": int(dual_metrics["llm_samples"]),
            "llm_calls": int(llm_metrics["llm_calls"]),
            "llm_top1_hits": int(llm_metrics["llm_top1_hits"]),
            "llm_top1_conditional": float(llm_metrics["llm_top1_conditional"]),
        }

        logger.info(
            f"[PIPE] dataset={dataset_name} span_top1={pipeline_span_top1:.4f} | "
            f"CORPUS_SET(P/R/F1)={pipe_corpus['precision']:.4f}/{pipe_corpus['recall']:.4f}/{pipe_corpus['f1']:.4f} "
            f"(TP/FP/FN={pipe_corpus['tp']}/{pipe_corpus['fp']}/{pipe_corpus['fn']}, "
            f"|G|={pipe_corpus['gold_set_size']}, |P|={pipe_corpus['pred_set_size']})"
        )

        # update global aggregation
        global_total_spans += total_spans
        global_dual_top1_hits += int(dual_metrics["dual_top1_hits"])
        global_dual_recallK_hits += int(dual_metrics["dual_recallK_hits"])
        global_pipeline_top1_hits_over_all_spans += int(pipeline_span_top1_hits_over_all)

        global_llm_samples += int(dual_metrics["llm_samples"])
        global_llm_calls += int(llm_metrics["llm_calls"])
        global_llm_top1_hits += int(llm_metrics["llm_top1_hits"])

        global_span_rows.extend(span_rows)

    # ---------- GLOBAL metrics ----------
    global_dual_span_top1 = global_dual_top1_hits / max(1, global_total_spans)
    global_dual_recallK = global_dual_recallK_hits / max(1, global_total_spans)
    global_pipeline_span_top1 = global_pipeline_top1_hits_over_all_spans / max(1, global_total_spans)
    global_llm_cond = (global_llm_top1_hits / global_llm_calls) if global_llm_calls > 0 else 0.0

    global_dual_corpus = corpus_set_prf(global_span_rows, pred_field="dual_pred_id")
    global_pipe_corpus = corpus_set_prf(global_span_rows, pred_field="pipeline_pred_id")

    results_summary["_GLOBAL"] = {
        "total_spans": int(global_total_spans),

        "dual_span_top1": float(global_dual_span_top1),
        "dual_recallK": float(global_dual_recallK),

        "pipeline_span_top1": float(global_pipeline_span_top1),

        # corpus-set
        "dual_corpus_tp": int(global_dual_corpus["tp"]),
        "dual_corpus_fp": int(global_dual_corpus["fp"]),
        "dual_corpus_fn": int(global_dual_corpus["fn"]),
        "dual_corpus_precision": float(global_dual_corpus["precision"]),
        "dual_corpus_recall": float(global_dual_corpus["recall"]),
        "dual_corpus_f1": float(global_dual_corpus["f1"]),
        "dual_gold_set_size": int(global_dual_corpus["gold_set_size"]),
        "dual_pred_set_size": int(global_dual_corpus["pred_set_size"]),

        "pipeline_corpus_tp": int(global_pipe_corpus["tp"]),
        "pipeline_corpus_fp": int(global_pipe_corpus["fp"]),
        "pipeline_corpus_fn": int(global_pipe_corpus["fn"]),
        "pipeline_corpus_precision": float(global_pipe_corpus["precision"]),
        "pipeline_corpus_recall": float(global_pipe_corpus["recall"]),
        "pipeline_corpus_f1": float(global_pipe_corpus["f1"]),
        "pipeline_gold_set_size": int(global_pipe_corpus["gold_set_size"]),
        "pipeline_pred_set_size": int(global_pipe_corpus["pred_set_size"]),

        "llm_samples": int(global_llm_samples),
        "llm_calls": int(global_llm_calls),
        "llm_top1_hits": int(global_llm_top1_hits),
        "llm_top1_conditional": float(global_llm_cond),
    }

    logger.info(
        "[GLOBAL] span: Dual top1=%.4f recall@%d=%.4f | Pipe top1=%.4f | LLM cond=%.4f | "
        "CORPUS_SET: Dual P/R/F1=%.4f/%.4f/%.4f (TP/FP/FN=%d/%d/%d, |G|=%d, |P|=%d) | "
        "Pipe P/R/F1=%.4f/%.4f/%.4f (TP/FP/FN=%d/%d/%d, |G|=%d, |P|=%d)",
        results_summary["_GLOBAL"]["dual_span_top1"],
        args.topk,
        results_summary["_GLOBAL"]["dual_recallK"],
        results_summary["_GLOBAL"]["pipeline_span_top1"],
        results_summary["_GLOBAL"]["llm_top1_conditional"],
        results_summary["_GLOBAL"]["dual_corpus_precision"],
        results_summary["_GLOBAL"]["dual_corpus_recall"],
        results_summary["_GLOBAL"]["dual_corpus_f1"],
        results_summary["_GLOBAL"]["dual_corpus_tp"],
        results_summary["_GLOBAL"]["dual_corpus_fp"],
        results_summary["_GLOBAL"]["dual_corpus_fn"],
        results_summary["_GLOBAL"]["dual_gold_set_size"],
        results_summary["_GLOBAL"]["dual_pred_set_size"],
        results_summary["_GLOBAL"]["pipeline_corpus_precision"],
        results_summary["_GLOBAL"]["pipeline_corpus_recall"],
        results_summary["_GLOBAL"]["pipeline_corpus_f1"],
        results_summary["_GLOBAL"]["pipeline_corpus_tp"],
        results_summary["_GLOBAL"]["pipeline_corpus_fp"],
        results_summary["_GLOBAL"]["pipeline_corpus_fn"],
        results_summary["_GLOBAL"]["pipeline_gold_set_size"],
        results_summary["_GLOBAL"]["pipeline_pred_set_size"],
    )

    # Save JSON
    summary_json_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON summary to {summary_json_path}")

    # Plots (span-level)
    plot_paths = plot_comparisons(results_summary, args.out_dir, topk=args.topk)

    # Markdown
    md_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPO DualLoRAEnc + LLM Refine Evaluation (Corpus-set PRF)\n\n")

        f.write("## Command\n\n```bash\n")
        f.write("python HPO_MoRE.py \\\n")
        for k, v in vars(args).items():
            if isinstance(v, list):
                for item in v:
                    f.write(f"  --{k} {item} \\\n")
            else:
                f.write(f"  --{k} {v} \\\n")
        f.write("```\n\n")

        g = results_summary["_GLOBAL"]
        f.write("## Global Metrics\n\n")
        f.write(f"- Total spans: **{g['total_spans']}**\n\n")

        f.write("### Span-level\n\n")
        f.write(f"- Dual span top-1: **{g['dual_span_top1']:.4f}**\n")
        f.write(f"- Dual recall@{args.topk}: **{g['dual_recallK']:.4f}**\n")
        f.write(f"- Pipeline span top-1: **{g['pipeline_span_top1']:.4f}**\n")
        f.write(f"- LLM conditional top-1 (called samples): **{g['llm_top1_conditional']:.4f}**\n\n")

        f.write("### Corpus-set (set PRF)\n\n")
        f.write(
            f"- Dual corpus-set P/R/F1: **{g['dual_corpus_precision']:.4f} / {g['dual_corpus_recall']:.4f} / {g['dual_corpus_f1']:.4f}** "
            f"(TP/FP/FN={g['dual_corpus_tp']}/{g['dual_corpus_fp']}/{g['dual_corpus_fn']}, |G|={g['dual_gold_set_size']}, |P|={g['dual_pred_set_size']})\n"
        )
        f.write(
            f"- Pipeline corpus-set P/R/F1: **{g['pipeline_corpus_precision']:.4f} / {g['pipeline_corpus_recall']:.4f} / {g['pipeline_corpus_f1']:.4f}** "
            f"(TP/FP/FN={g['pipeline_corpus_tp']}/{g['pipeline_corpus_fp']}/{g['pipeline_corpus_fn']}, |G|={g['pipeline_gold_set_size']}, |P|={g['pipeline_pred_set_size']})\n\n"
        )

        f.write("## Per-dataset Metrics\n\n")
        f.write(
            "| Dataset | Spans | Dual span top1 | Dual recall@K | Pipe span top1 | "
            "Dual corpus P | Dual corpus R | Dual corpus F1 | Dual TP/FP/FN | "
            "Pipe corpus P | Pipe corpus R | Pipe corpus F1 | Pipe TP/FP/FN | LLM calls | LLM cond |\n"
        )
        f.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|\n"
        )

        for ds_name, m in results_summary.items():
            if ds_name.startswith("_"):
                continue
            f.write(
                f"| {ds_name} | {m['total_spans']} | "
                f"{m['dual_span_top1']:.4f} | {m['dual_recallK']:.4f} | {m['pipeline_span_top1']:.4f} | "
                f"{m['dual_corpus_precision']:.4f} | {m['dual_corpus_recall']:.4f} | {m['dual_corpus_f1']:.4f} | "
                f"{m['dual_corpus_tp']}/{m['dual_corpus_fp']}/{m['dual_corpus_fn']} | "
                f"{m['pipeline_corpus_precision']:.4f} | {m['pipeline_corpus_recall']:.4f} | {m['pipeline_corpus_f1']:.4f} | "
                f"{m['pipeline_corpus_tp']}/{m['pipeline_corpus_fp']}/{m['pipeline_corpus_fn']} | "
                f"{m['llm_calls']} | {m['llm_top1_conditional']:.4f} |\n"
            )

        f.write("\n## Plots (span-level)\n\n")
        if plot_paths.get("top1"):
            f.write(f"![Span Top-1 Comparison]({os.path.basename(plot_paths['top1'])})\n\n")
        if plot_paths.get("recall_llm"):
            f.write(f"![Recall vs LLM Conditional]({os.path.basename(plot_paths['recall_llm'])})\n\n")

        f.write(
            "\n### Corpus-set meaning\n"
            "- We treat each **corpus** as a phenotype **set**.\n"
            "- TP/FP/FN are computed on **sets** (unique HPO IDs over the corpus), not spans.\n"
        )

    logger.info(f"Markdown summary saved to: {md_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
