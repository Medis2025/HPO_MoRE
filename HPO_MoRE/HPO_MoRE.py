#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPO_MoRE.py

DualLoRAEnc + LLM refine eval for HPO-ID with margin-based gating:

Step 1: DualLoRAEnc (span encoder) does ALL-HPO retrieval
        on each dataset (GeneReviews / GSC+ / ID-68), using a **global**
        HPO embedding table (all HPO IDs in the ontology, not just those
        appearing in the dataset). For each dataset:
          - computes top-1 accuracy
          - computes recall@K (K = --topk, default 15)
          - builds top-K candidate lists for each span (with Dual scores)

Step 2: For spans where the gold HPO is in top-K, apply margin-based gating:

    Let margin = score_top1 - score_top2 (DualLoRAEnc similarity margin).

    - If margin >= tau_high   (easy case):
        -> Trust DualLoRAEnc top-1 directly, DO NOT call LLM.

    - If margin <= tau_low    (hard case):
        -> Call LLM and trust its top choice among the K candidates.

    - If tau_low < margin < tau_high (medium case):
        -> Call LLM:
            * If LLM hits gold, accept LLM.
            * If LLM misses, fall back to DualLoRAEnc top-1.

Thus the final prediction is a hybrid of DualLoRAEnc + LLM,
with LLM mainly used on "hard" or ambiguous cases.

Additionally:
  - Compute **global** metrics across all datasets:
      * DualLoRAEnc global top-1
      * DualLoRAEnc global recall@K
      * LLM global conditional top-1 (only on samples where LLM was actually called)
      * Full pipeline global top-1 (Dual+LLM with margin gating)
  - Save:
      * JSON summary
      * Markdown report with comparison plots:
          - top1_comparison.png : Dual vs Pipeline per dataset
          - recall_vs_llm.png   : Recall@K vs LLM conditional top-1 per dataset

Example usage:

python HPO_MoRE.py \
  --eval_roots \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GeneReviews \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GSC+ \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/ID-68 \
  --val_root /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/val \
  --hpo_json /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/hpo.json \
  --model_dir /cluster/home/gw/Backend_project/NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --init_encoder_from /cluster/home/gw/Backend_project/NER/tuned/intention \
  --span_ckpt /cluster/home/gw/Backend_project/NER/tuned/hpoid_span_contrastive/hpoid_span_best.pt \
  --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/VAL_LLM_35 \
  --prompt_path /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/prompts/select_hpo.txt \
  --api_key_env DEEPSEEK_API_KEY \
  --topk 35 \
  --num_workers 16 \
  --tau_low 0.05 \
  --tau_high 0.20 \
  --hpo_chunk_size 512

NOTE:
  - DO NOT hardcode API keys in this script.
    Use an environment variable and specify its name via --api_key_env, e.g.:

      export DEEPSEEK_API_KEY="sk-xxxx"
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple

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
# Helper: extract HPO info for prompt
# -------------------------------------------------------------------------
def get_hpo_prompt_info(ontology: HPOOntology, hpo_id: str) -> Dict[str, Any]:
    """
    Map HPOOntology data into a light structure for LLM:

      {
        "hpo_id": "HP:0001250",
        "hpo_name": "...",
        "hpo_def": "...",
        "hpo_synonyms": [...],
      }
    """
    hid = ontology.resolve_id(hpo_id)
    rec = ontology.data.get(hid, {}) or {}

    # Name
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

    # Definition
    d = rec.get("Def") or rec.get("def") or ""
    if isinstance(d, list):
        d = d[0] if d else ""
    if not isinstance(d, str):
        d = ""

    # Synonyms
    syns = rec.get("Synonym") or rec.get("synonym") or []
    if isinstance(syns, str):
        syns = [syns]
    elif not isinstance(syns, list):
        syns = []

    return {
        "hpo_id": hid,
        "hpo_name": name,
        "hpo_def": d,
        "hpo_synonyms": syns,
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
    """
    Build a **global** HPO embedding table over ALL terms in the ontology.

    - Takes all ontology.data keys as candidate HPO IDs.
    - Encodes them in chunks to avoid OOM:
        * For each chunk of HPO IDs, call encode_hpo_gold_table(...)
        * Collect all embeddings + valid HPO IDs
    - Concatenates to z_hpo_full [N, D] on `device`
    - Builds id2idx mapping so dataset gold IDs can be located in the global table.

    Returns
    -------
    z_hpo_full : torch.Tensor
        [N, D] global HPO embedding matrix.
    hpo_ids_full : list of str
        HPO IDs aligned with rows in z_hpo_full.
    id2idx_full : dict
        Mapping from HPO ID -> row index in z_hpo_full.
    """
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
# (Legacy) Dataset-specific HPO table (no longer used; kept for reference)
# -------------------------------------------------------------------------
def build_dataset_hpo_table(
    ds: HPOIDSpanPairDataset,
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer,
    ontology: HPOOntology,
    cfg: HPOConfig,
    device: torch.device,
):
    """
    (Unused in the global-table setting)
    For the current dataset, build an HPO gold table restricted to HPO IDs
    that actually appear in this dataset.

      - hpo_ids_table: HPO IDs that appear as gold in ds and exist in ontology
      - encode_hpo_gold_table -> (z_hpo, hpo_ids_vec)
    """
    hpo_ids_table = sorted({ex["hpo_id"] for ex in ds if ex["hpo_id"] in ontology.data})
    logger.info(f"[DualLoRAEnc] dataset has {len(hpo_ids_table)} unique HPO IDs (subset table).")

    z_hpo, hpo_ids_vec = encode_hpo_gold_table(
        model_tc,
        span_proj,
        tokenizer,
        ontology,
        hpo_ids_table,
        device=device,
        max_len=cfg.max_len,
    )
    if z_hpo is None or z_hpo.numel() == 0:
        raise RuntimeError("[DualLoRAEnc] HPO embedding table is empty.")

    logger.info(
        f"[DualLoRAEnc] HPO table size = {z_hpo.size(0)}, dim = {z_hpo.size(1)}, device={z_hpo.device}"
    )

    id2idx = {hid: i for i, hid in enumerate(hpo_ids_vec)}
    return z_hpo, hpo_ids_vec, id2idx


# -------------------------------------------------------------------------
# DualLoRAEnc: build top-K candidates + basic metrics (with margin)
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
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    For the entire dataset:

    - Use DualLoRAEnc (z_left @ z_hpo^T) to do HPO retrieval against the
      provided HPO table (which in this script is the **global** HPO table).
    - Compute:
        * dual_top1        : gold is the top-1 prediction
        * dual_recallK     : gold appears within the top-K predictions
        * dual_top1_hits   : number of correct top-1 predictions
        * dual_recallK_hits: number of spans where gold ∈ top-K
    - Build LLM input samples ONLY for spans where gold ∈ top-K:

        samples_for_llm = [
          {
            "dataset": dataset_name,
            "idx": global_index,
            "context": left_text (truncated),
            "mention": mention_text,
            "gold_id": gold_hpo,
            "candidates": [      # sorted by Dual score desc
                {
                  "hpo_id": ...,
                  "hpo_name": ...,
                  "hpo_def": ...,
                  "hpo_synonyms": [...],
                  "score": float,
                }, * topK
            ],
            "dual_best_id": hpo_id of top1,
            "dual_margin": score_top1 - score_top2 (or 0 if no top2),
          }, ...
        ]
    """

    BATCH = cfg.batch_size
    dual_top1_hits = 0
    dual_recallK_hits = 0
    total = 0

    samples_for_llm: List[Dict[str, Any]] = []

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

        # Encode mention spans
        z_left = encode_spans(
            model_tc,
            span_proj,
            tokenizer,
            left_texts,
            left_spans,
            device,
            cfg.max_len,
        )  # [b, D]

        # Retrieval against provided HPO table: z_left @ z_hpo^T
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

            # Index of gold in global HPO table
            gold_idx = id2idx.get(gold, None)

            preds = idxs[row].tolist()

            # DualLoRAEnc top-1
            if gold_idx is not None and preds:
                if preds[0] == gold_idx:
                    dual_top1_hits += 1

            # recall@K
            gold_in_topk = False
            if gold_idx is not None and gold_idx in preds:
                dual_recallK_hits += 1
                gold_in_topk = True

            # Only construct LLM samples for spans where gold ∈ topK
            if gold_in_topk:
                cand_list = []
                for rank_pos in range(inner_topk):
                    idx_hpo = idxs[row, rank_pos].item()
                    score = float(vals[row, rank_pos].item())
                    hid = hpo_ids_vec[idx_hpo]
                    info = get_hpo_prompt_info(ontology, hid)
                    info["score"] = score
                    cand_list.append(info)

                # top1 / margin
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

                # Optional truncation of context
                if len(left_text) > 512:
                    context = left_text[:512]
                else:
                    context = left_text

                sample = {
                    "dataset": dataset_name,
                    "idx": global_idx,
                    "context": context,
                    "mention": mention_text,
                    "gold_id": gold,
                    "candidates": cand_list,
                    "dual_best_id": dual_best_id,
                    "dual_margin": float(margin),
                }
                samples_for_llm.append(sample)

            global_idx += 1

    dual_top1 = dual_top1_hits / max(1, total)
    dual_recallK = dual_recallK_hits / max(1, total)

    logger.info(
        f"[DualLoRAEnc] dataset={dataset_name} top1={dual_top1:.4f}, "
        f"recall@{topk}={dual_recallK:.4f}, total_spans={total}, "
        f"LLM_samples={len(samples_for_llm)}"
    )

    metrics = {
        "dual_top1": float(dual_top1),
        "dual_recallK": float(dual_recallK),
        "dual_top1_hits": int(dual_top1_hits),
        "dual_recallK_hits": int(dual_recallK_hits),
        "total_spans": int(total),
        "llm_samples": int(len(samples_for_llm)),
    }
    return samples_for_llm, metrics


# -------------------------------------------------------------------------
# LLM refine with margin-based gating + multithreading + progress bar
# -------------------------------------------------------------------------
def run_llm_refine_for_dataset(
    dataset_name: str,
    samples_for_llm: List[Dict[str, Any]],
    refiner: HPOCandidateRefiner,
    num_workers: int = 16,
    tau_low: float = 0.05,
    tau_high: float = 0.20,
) -> Dict[str, float]:
    """
    Run LLM refinement with margin-based gating on a dataset:

      For each sample (gold ∈ topK):

        margin = dual_margin
        dual_best_id = sample["dual_best_id"]

        If margin >= tau_high (easy case):
            -> Prediction = dual_best_id
            -> NO LLM CALL

        If margin <= tau_low (hard case):
            -> CALL LLM:
                 indices = refiner.refine(context, mention, candidates)
               If indices valid:
                 pred = candidates[indices[0]]
               Else:
                 pred = dual_best_id (fallback)
            -> LLM is considered "called" here.

        If tau_low < margin < tau_high (medium case):
            -> CALL LLM:
                 if LLM prediction == gold: use LLM
                 else: fallback to dual_best_id
            -> LLM is considered "called" here.

    We track:
      - pipeline_top1_hits: final hits of the combined pipeline
      - llm_calls: number of samples where we actually invoked LLM
      - llm_top1_hits: number of hits when using LLM's own choice
        (for conditional accuracy: hits / llm_calls)

    Returns:
      {
        "pipeline_top1": float,           # pipeline hits / n_samples
        "pipeline_top1_hits": int,
        "n_samples": int,                 # number of samples_for_llm
        "llm_calls": int,
        "llm_top1_hits": int,
        "llm_top1_conditional": float,    # hits / llm_calls (0 if llm_calls==0)
      }
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
        }

    logger.info(
        f"[LLM] Starting margin-gated refine on dataset={dataset_name} with {len(samples_for_llm)} samples..."
    )

    n = len(samples_for_llm)

    def _run_one(sample: Dict[str, Any]):
        """
        Returns:
          (pipeline_hit: bool, llm_called: bool, llm_hit: bool)
        """
        context = sample["context"]
        mention = sample["mention"]
        gold_id = sample["gold_id"]
        candidates = sample["candidates"]
        dual_best_id = sample.get("dual_best_id", None)
        margin = float(sample.get("dual_margin", 0.0))

        # If we don't even have dual_best_id, we cannot do much; treat as miss
        if dual_best_id is None:
            return False, False, False

        # Easy case: high margin -> trust Dual, no LLM call
        if margin >= tau_high:
            pipeline_hit = (dual_best_id == gold_id)
            return pipeline_hit, False, False

        # Hard / medium: we will call LLM
        llm_called = True
        llm_hit = False
        pred_final = dual_best_id  # fallback

        try:
            indices = refiner.refine(context, mention, candidates)
        except Exception as e:
            logger.warning(f"[LLM] Error in refine: {e}")
            # Fallback: keep pred_final = dual_best_id
            pipeline_hit = (pred_final == gold_id)
            return pipeline_hit, llm_called, llm_hit

        if not indices:
            # LLM returns empty / -1; treat as miss for LLM, fallback Dual
            pipeline_hit = (pred_final == gold_id)
            return pipeline_hit, llm_called, llm_hit

        pred_idx = indices[0]
        if pred_idx < 0 or pred_idx >= len(candidates):
            # invalid index
            pipeline_hit = (pred_final == gold_id)
            return pipeline_hit, llm_called, llm_hit

        llm_hid = candidates[pred_idx]["hpo_id"]
        llm_hit = (llm_hid == gold_id)

        if margin <= tau_low:
            # Hard case: fully trust LLM
            pred_final = llm_hid
            pipeline_hit = (pred_final == gold_id)
            return pipeline_hit, llm_called, llm_hit

        # Medium case: LLM can override if correct, else fallback Dual
        if llm_hit:
            pred_final = llm_hid
        else:
            pred_final = dual_best_id

        pipeline_hit = (pred_final == gold_id)
        return pipeline_hit, llm_called, llm_hit

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
                pipeline_hit, llm_called, llm_hit = fut.result()
            except Exception as e:
                logger.warning(f"[LLM] Future error: {e}")
                continue

            if pipeline_hit:
                pipeline_hits += 1
            if llm_called:
                llm_calls += 1
                if llm_hit:
                    llm_hits += 1

    pipeline_top1 = pipeline_hits / max(1, n)
    if llm_calls > 0:
        llm_cond = llm_hits / llm_calls
    else:
        llm_cond = 0.0

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
    }


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------
def plot_comparisons(
    results_summary: Dict[str, Dict[str, Any]],
    out_dir: str,
    topk: int,
) -> Dict[str, str]:
    """
    Generate comparison plots:

      - top1_comparison.png:
          For each dataset: DualLoRAEnc top1 vs Pipeline refined top1
      - recall_vs_llm.png:
          For each dataset: DualLoRAEnc recall@K vs LLM conditional top-1

    Returns:
      { "top1": <path or "">, "recall_llm": <path or ""> }
    """
    if not HAS_MPL:
        logger.warning("[PLOT] matplotlib not available, skip plotting.")
        return {"top1": "", "recall_llm": ""}

    # Exclude any global rows (key starting with "_")
    dataset_names = [
        k for k in results_summary.keys() if not k.startswith("_")
    ]
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

    # 1) Top-1 comparison
    plt.figure(figsize=(fig_w, 4.5))
    plt.bar(
        [xi - width / 2 for xi in x],
        dual_top1_vals,
        width=width,
        label="DualLoRAEnc Top-1",
    )
    plt.bar(
        [xi + width / 2 for xi in x],
        pipe_top1_vals,
        width=width,
        label="Pipeline (Dual+LLM gating) Top-1",
    )
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Top-1 Accuracy")
    plt.title("Top-1 Comparison: DualLoRAEnc vs Dual+LLM (margin-gated)")
    plt.legend()
    plt.tight_layout()

    top1_path = os.path.join(out_dir, "top1_comparison.png")
    plt.savefig(top1_path, dpi=200)
    plt.close()
    logger.info(f"[PLOT] Saved top-1 comparison to {top1_path}")

    # 2) Recall@K vs LLM conditional
    plt.figure(figsize=(fig_w, 4.5))
    plt.bar(
        [xi - width / 2 for xi in x],
        recall_vals,
        width=width,
        label=f"DualLoRAEnc Recall@{topk}",
    )
    plt.bar(
        [xi + width / 2 for xi in x],
        llm_cond_vals,
        width=width,
        label="LLM Conditional Top-1 (on called samples)",
    )
    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Ratio")
    plt.title(f"Recall@{topk} vs LLM Conditional Top-1")
    plt.legend()
    plt.tight_layout()

    recall_path = os.path.join(out_dir, "recall_vs_llm.png")
    plt.savefig(recall_path, dpi=200)
    plt.close()
    logger.info(f"[PLOT] Saved recall vs LLM comparison to {recall_path}")

    return {"top1": top1_path, "recall_llm": recall_path}


# -------------------------------------------------------------------------
# CLI & main
# -------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="DualLoRAEnc eval + LLM refine for HPO-ID (topK candidates) with margin-based gating (using global HPO table)."
    )
    ap.add_argument(
        "--eval_roots",
        type=str,
        nargs="+",
        required=True,
        help="Eval roots (GeneReviews / GSC+ / ID-68), each with ann/ and corpus/.",
    )
    ap.add_argument(
        "--val_root",
        type=str,
        default=None,
        help="Optional extra validation root; if set, it is evaluated as an additional dataset.",
    )
    ap.add_argument(
        "--hpo_json",
        type=str,
        required=True,
        help="Path to hpo.json",
    )
    ap.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="NER LoRA model dir (TokenCRFWrapper / PeftModel).",
    )
    ap.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="HF backbone path (e.g., BioLinkBERT-base).",
    )
    ap.add_argument(
        "--init_encoder_from",
        type=str,
        default=None,
        help="Optional encoder init checkpoint (e.g., intention NER ckpt).",
    )
    ap.add_argument(
        "--span_ckpt",
        type=str,
        required=True,
        help="Path to span projection checkpoint (hpoid_span_best.pt).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for eval json, markdown and plots.",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=15, help="topK candidates for LLM refine.")
    # LLM related
    ap.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to prompt txt file (used by LLMAPIClient).",
    )
    ap.add_argument(
        "--api_key_env",
        type=str,
        default="DEEPSEEK_API_KEY",
        help="Env var name for LLM API key.",
    )
    ap.add_argument(
        "--base_url",
        type=str,
        default="https://api.deepseek.com",
        help="Base URL of LLM API.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="LLM model name.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Max threads for LLM refine.",
    )
    # margin gating thresholds
    ap.add_argument(
        "--tau_low",
        type=float,
        default=0.05,
        help="Margin threshold for HARD cases (margin <= tau_low -> trust LLM).",
    )
    ap.add_argument(
        "--tau_high",
        type=float,
        default=0.20,
        help="Margin threshold for EASY cases (margin >= tau_high -> trust Dual).",
    )
    # global HPO table chunk size
    ap.add_argument(
        "--hpo_chunk_size",
        type=int,
        default=512,
        help="Chunk size when encoding the global HPO table (to avoid OOM).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Basic HPOConfig (minimal fields needed here)
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

    # Ontology
    ontology = HPOOntology(args.hpo_json)
    logger.info(f"Loaded HPO ontology with {len(ontology.data)} nodes from {args.hpo_json}.")

    # Tokenizer + NER encoder
    tokenizer, model_tc, meta = load_ner_tc_and_tokenizer(
        args.backbone,
        args.init_encoder_from,
        args.model_dir,
        cfg,
    )
    model_tc.to(device)
    model_tc.eval()
    hidden_size = model_tc.base.config.hidden_size

    # Span projection head
    if not os.path.isfile(args.span_ckpt):
        raise FileNotFoundError(f"Span checkpoint not found: {args.span_ckpt}")
    ckpt = torch.load(args.span_ckpt, map_location="cpu")
    span_dim = ckpt.get("cfg", {}).get("hpoid_dim", 256)
    span_proj = SpanProj(in_dim=hidden_size, out_dim=span_dim, dropout=0.0).to(device)
    span_proj.load_state_dict(ckpt["span_proj_state"])
    span_proj.eval()
    logger.info(
        f"Loaded span projection head from {args.span_ckpt} (epoch={ckpt.get('epoch','?')})."
    )

    # LLM client + refiner
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

    # Save eval config
    cfg_out = {
        "cli": vars(args),
        "meta": meta,
    }
    cfg_path = os.path.join(args.out_dir, "hpo_revise_llm_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved config to {cfg_path}")

    # ------------------------------------------------------------------
    # Build a **single global HPO table** over ALL ontology IDs (once)
    # ------------------------------------------------------------------
    z_hpo_global, hpo_ids_global, id2idx_global = build_global_hpo_table(
        model_tc=model_tc,
        span_proj=span_proj,
        tokenizer=tokenizer,
        ontology=ontology,
        cfg=cfg,
        device=device,
        chunk_size=args.hpo_chunk_size,
    )

    # results_summary[dataset] = {...}
    results_summary: Dict[str, Dict[str, Any]] = {}

    # Global counters for overall metrics
    total_spans_all = 0
    dual_top1_hits_all = 0
    dual_recall_hits_all = 0

    # For pipeline & LLM
    llm_samples_all = 0          # total samples with gold ∈ topK
    pipeline_hits_all = 0        # final pipeline hits among samples_for_llm
    llm_calls_all = 0
    llm_top1_hits_all = 0

    # Collect eval roots (+ optional val_root)
    eval_roots = list(args.eval_roots)
    if args.val_root:
        eval_roots.append(args.val_root)

    # Evaluate each dataset root separately
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

        # 1) Use the GLOBAL HPO table for retrieval
        z_hpo = z_hpo_global
        hpo_ids_vec = hpo_ids_global
        id2idx = id2idx_global

        # 2) DualLoRAEnc: build top-K candidates & basic metrics (over global table)
        samples_for_llm, dual_metrics = build_candidates_with_duallora(
            dataset_name,
            ds,
            model_tc,
            span_proj,
            tokenizer,
            ontology,
            cfg,
            device,
            z_hpo,
            hpo_ids_vec,
            id2idx,
            topk=args.topk,
        )

        # 3) LLM refine with margin-based gating (only on gold ∈ topK samples)
        llm_metrics = run_llm_refine_for_dataset(
            dataset_name,
            samples_for_llm,
            refiner,
            num_workers=args.num_workers,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
        )

        # Pipeline top-1 for this dataset:
        # pipeline_hits among llm_samples, divided by total spans
        if dual_metrics["total_spans"] > 0:
            pipeline_top1 = llm_metrics["pipeline_top1_hits"] / dual_metrics["total_spans"]
        else:
            pipeline_top1 = 0.0

        # Record per-dataset metrics
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

        # Update global counters
        total_spans_all += dual_metrics["total_spans"]
        dual_top1_hits_all += dual_metrics["dual_top1_hits"]
        dual_recall_hits_all += dual_metrics["dual_recallK_hits"]

        llm_samples_all += dual_metrics["llm_samples"]
        pipeline_hits_all += llm_metrics["pipeline_top1_hits"]
        llm_calls_all += llm_metrics["llm_calls"]
        llm_top1_hits_all += llm_metrics["llm_top1_hits"]

    # Compute global metrics across all datasets
    if total_spans_all > 0:
        global_dual_top1 = dual_top1_hits_all / total_spans_all
        global_dual_recallK = dual_recall_hits_all / total_spans_all
        global_pipeline_top1 = pipeline_hits_all / total_spans_all
    else:
        global_dual_top1 = 0.0
        global_dual_recallK = 0.0
        global_pipeline_top1 = 0.0

    if llm_calls_all > 0:
        global_llm_conditional = llm_top1_hits_all / llm_calls_all
    else:
        global_llm_conditional = 0.0

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
        "[GLOBAL] DualLoRAEnc top1=%.4f, recall@%d=%.4f, "
        "LLM conditional top1=%.4f (on called samples), Pipeline top1=%.4f (total_spans=%d)",
        global_dual_top1,
        args.topk,
        global_dual_recallK,
        global_llm_conditional,
        global_pipeline_top1,
        total_spans_all,
    )

    # Save JSON summary
    summary_json_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON summary to {summary_json_path}")

    # Plot comparisons
    plot_paths = plot_comparisons(results_summary, args.out_dir, topk=args.topk)

    # Save Markdown summary with plots
    md_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPO DualLoRAEnc + LLM Refine Evaluation (Margin-Gated Pipeline, Global HPO Table)\n\n")

        f.write("## Command\n\n```bash\n")
        f.write("python HPO_MoRE.py \\\n")
        for k, v in vars(args).items():
            if isinstance(v, list):
                for item in v:
                    f.write(f"  --{k} {item} \\\n")
            else:
                f.write(f"  --{k} {v} \\\n")
        f.write("```\n\n")

        # Global metrics
        g = results_summary.get("_GLOBAL", {})
        f.write("## Global Metrics (All Datasets Combined)\n\n")
        f.write(f"- Total spans: **{g.get('total_spans', 0)}**\n")
        f.write("- DualLoRAEnc Top-1 (global HPO table): **{:.4f}**\n".format(g.get("dual_top1", 0.0)))
        f.write(
            "- DualLoRAEnc Recall@{} (global HPO table): **{:.4f}**\n".format(
                args.topk, g.get("dual_recallK", 0.0)
            )
        )
        f.write(
            "- LLM Conditional Top-1 (on called samples, given gold ∈ top-{}): "
            "**{:.4f}**\n".format(args.topk, g.get("llm_top1_conditional", 0.0))
        )
        f.write(
            "- **Full Pipeline Top-1 (DualLoRAEnc + LLM refine with gating, global HPO table)**: "
            "**{:.4f}**\n\n".format(g.get("pipeline_top1", 0.0))
        )

        # Per-dataset table
        f.write("## Metrics per Dataset\n\n")
        f.write(
            "| Dataset | Dual Top-1 | Dual Recall@{} | Total spans | LLM samples | LLM calls | LLM Top-1 (cond.) | Pipeline Top-1 |\n".format(
                args.topk
            )
        )
        f.write(
            "|---------|-----------:|---------------:|------------:|------------:|----------:|------------------:|---------------:|\n"
        )

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

        # Plots
        f.write("\n## Plots\n\n")
        if plot_paths.get("top1"):
            f.write(
                f"![Top-1 Comparison]({os.path.basename(plot_paths['top1'])})\n\n"
            )
        if plot_paths.get("recall_llm"):
            f.write(
                f"![Recall@{args.topk} vs LLM Conditional Top-1]({os.path.basename(plot_paths['recall_llm'])})\n\n"
            )

        f.write(
            "\nThis report summarizes the performance of pure DualLoRAEnc retrieval on a **global** HPO table "
            "and the margin-gated pipeline that combines DualLoRAEnc top-K retrieval with LLM-based "
            "candidate selection, using the Dual similarity margin to decide when to trust "
            "the encoder vs when to rely on the LLM.\n"
        )

    logger.info(f"Markdown summary saved to: {md_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
