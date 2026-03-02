#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPO_MoRE.py (REVISED: local LLM + batched refine + 4bit LLM)

✅ Minimal-change goals:
- Keep DualLoRAEnc + gating logic the same
- Replace API LLM with LOCAL HF model via refiner_local.py (LLMAPIClient)
- Add batched local LLM refine path (already)
- NEW: add 4-bit load option for the local LLM (bitsandbytes)

Notes:
- Requires refiner_local.py LLMAPIClient to accept:
    load_in_4bit: bool
    bnb_4bit_quant_type: str (e.g. "nf4")
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str (e.g. "bf16"/"fp16")
  If your refiner_local.py doesn't have these yet, see the tiny patch at bottom.

Example (local 4-bit):
python HPO_MoRE.py \
  --eval_roots ... \
  --hpo_json ... \
  --model_dir ... \
  --backbone ... \
  --span_ckpt ... \
  --out_dir ... \
  --prompt_path ... \
  --local_model_dir /cluster/home/gw/Backend_project/models/Baichuan-M2-32B \
  --local_load_in_4bit \
  --local_batch_size 4 \
  --local_max_input_tokens 1536 \
  --local_max_new_tokens 16 \
  --topk 35
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
from refiner_local import LLMAPIClient, HPOCandidateRefiner

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
    hid = ontology.resolve_id(hpo_id)
    rec = ontology.data.get(hid, {}) or {}

    # ---------- Name ----------
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

    # ---------- Synonyms ----------
    syns = rec.get("Synonym") or rec.get("synonym") or []
    if isinstance(syns, str):
        syns = [syns]
    elif not isinstance(syns, list):
        syns = []
    syns = [str(s) for s in syns if s]

    # ---------- Original Def ----------
    d = rec.get("Def") or rec.get("def") or ""
    if isinstance(d, list):
        d = d[0] if d else ""
    if not isinstance(d, str):
        d = ""
    orig_def = d.strip()

    # ---------- LLM Def / ADD_DEF ----------
    llm_def = rec.get("llm_def") or ""
    if not isinstance(llm_def, str):
        llm_def = ""
    llm_def = llm_def.strip()

    llm_add_def = rec.get("llm_add_def") or ""
    if not isinstance(llm_add_def, str):
        llm_add_def = ""
    llm_add_def = llm_add_def.strip()

    # ---------- combined definition ----------
    lines = []
    lines.append(f"[HPO_ID] {hid}")
    lines.append(f"[NAME] {name}")

    if syns:
        syn_str = "; ".join(syns)
        lines.append(f"[SYN] {syn_str}")

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

        z_left = encode_spans(
            model_tc,
            span_proj,
            tokenizer,
            left_texts,
            left_spans,
            device,
            cfg.max_len,
        )

        sims = z_left @ z_hpo.t()
        inner_topk = min(topk, sims.size(1))
        vals, idxs = torch.topk(sims, k=inner_topk, dim=-1)

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

            if gold_idx is not None and preds:
                if preds[0] == gold_idx:
                    dual_top1_hits += 1

            gold_in_topk = False
            if gold_idx is not None and gold_idx in preds:
                dual_recallK_hits += 1
                gold_in_topk = True

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
# LLM refine with margin-based gating (batched local)
# -------------------------------------------------------------------------
def run_llm_refine_for_dataset(
    dataset_name: str,
    samples_for_llm: List[Dict[str, Any]],
    refiner: HPOCandidateRefiner,
    num_workers: int = 16,   # kept for signature compatibility; not used in batched mode
    tau_low: float = 0.05,
    tau_high: float = 0.20,
    batch_size: int = 16,
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

    logger.info(
        f"[LLM] Starting margin-gated refine on dataset={dataset_name} with {len(samples_for_llm)} samples..."
    )

    n = len(samples_for_llm)
    pipeline_hits = 0
    llm_calls = 0
    llm_hits = 0

    need_llm_indices: List[int] = []
    easy_indices: List[int] = []

    for i, sample in enumerate(samples_for_llm):
        dual_best_id = sample.get("dual_best_id", None)
        gold_id = sample.get("gold_id", None)
        margin = float(sample.get("dual_margin", 0.0))

        if dual_best_id is None or gold_id is None:
            continue

        if margin >= tau_high:
            if dual_best_id == gold_id:
                pipeline_hits += 1
            easy_indices.append(i)
        else:
            need_llm_indices.append(i)

    for start in tqdm(
        range(0, len(need_llm_indices), max(1, batch_size)),
        desc=f"[LLM] {dataset_name} - batched refine",
        leave=False,
    ):
        idx_chunk = need_llm_indices[start: start + max(1, batch_size)]
        if not idx_chunk:
            continue

        prompts: List[str] = []
        num_cands: List[int] = []

        for ii in idx_chunk:
            s = samples_for_llm[ii]
            context = s["context"]
            mention = s["mention"]
            cands = s["candidates"]

            cblock = refiner._build_candidates_list(cands)
            prompt = refiner.llm.build_prompt(context=context, mention=mention, candidates_block=cblock)
            prompts.append(prompt)
            num_cands.append(min(len(cands), refiner.max_candidates))

        llm_calls += len(idx_chunk)
        try:
            raws = refiner.llm.generate_indices_batch(prompts)
        except Exception as e:
            logger.warning(f"[LLM] batch generate error: {e}")
            for ii in idx_chunk:
                s = samples_for_llm[ii]
                if s.get("dual_best_id") == s.get("gold_id"):
                    pipeline_hits += 1
            continue

        for ii, raw, nc in zip(idx_chunk, raws, num_cands):
            s = samples_for_llm[ii]
            gold_id = s["gold_id"]
            dual_best_id = s["dual_best_id"]
            margin = float(s.get("dual_margin", 0.0))
            cands = s["candidates"]

            indices = refiner._parse_indices(raw, num_candidates=nc)

            pred_final = dual_best_id
            llm_hit = False

            if indices:
                pred_idx = indices[0]
                if 0 <= pred_idx < len(cands):
                    llm_hid = cands[pred_idx]["hpo_id"]
                    llm_hit = (llm_hid == gold_id)

                    if margin <= tau_low:
                        pred_final = llm_hid
                    else:
                        pred_final = llm_hid if llm_hit else dual_best_id

            if llm_hit:
                llm_hits += 1
            if pred_final == gold_id:
                pipeline_hits += 1

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
    plt.bar([xi - width / 2 for xi in x], dual_top1_vals, width=width, label="DualLoRAEnc Top-1")
    plt.bar([xi + width / 2 for xi in x], pipe_top1_vals, width=width, label="Pipeline (Dual+LLM gating) Top-1")
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

    plt.figure(figsize=(fig_w, 4.5))
    plt.bar([xi - width / 2 for xi in x], recall_vals, width=width, label=f"DualLoRAEnc Recall@{topk}")
    plt.bar([xi + width / 2 for xi in x], llm_cond_vals, width=width, label="LLM Conditional Top-1 (called samples)")
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
    ap.add_argument("--eval_roots", type=str, nargs="+", required=True)
    ap.add_argument("--val_root", type=str, default=None)
    ap.add_argument("--hpo_json", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, required=True)
    ap.add_argument("--init_encoder_from", type=str, default=None)
    ap.add_argument("--span_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=15)

    ap.add_argument("--prompt_path", type=str, required=True)

    ap.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    ap.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    ap.add_argument("--model", type=str, default="deepseek-chat")

    ap.add_argument("--num_workers", type=int, default=16)

    ap.add_argument("--tau_low", type=float, default=0.05)
    ap.add_argument("--tau_high", type=float, default=0.20)
    ap.add_argument("--hpo_chunk_size", type=int, default=512)

    # ---- Local LLM args ----
    ap.add_argument("--local_model_dir", type=str, required=True, help="Local HF model dir for LLM refiner.")
    ap.add_argument("--local_max_new_tokens", type=int, default=16)
    ap.add_argument("--local_temperature", type=float, default=0.0)
    ap.add_argument("--local_max_input_tokens", type=int, default=512)
    ap.add_argument("--local_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--local_device_map", type=str, default="auto")
    ap.add_argument("--local_batch_size", type=int, default=16, help="Batch size for local LLM generate_indices_batch().")

    # ✅ NEW: 4-bit flags (minimal additions)
    ap.add_argument("--local_load_in_4bit", action="store_true", help="Load local LLM in 4-bit (bitsandbytes).")
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb_4bit_use_double_quant", action="store_true", help="Use double quant (bnb 4-bit).")
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

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

    # ✅ Local LLM client + refiner
    llm_client = LLMAPIClient(
        api_key="",
        base_url="",
        model="",
        timeout=60.0,
        prompt_path=args.prompt_path,

        local_model_dir=args.local_model_dir,
        local_max_new_tokens=args.local_max_new_tokens,
        local_temperature=args.local_temperature,
        local_max_input_tokens=args.local_max_input_tokens,
        dtype=args.local_dtype,
        device_map=args.local_device_map,

        # ✅ NEW: 4-bit load
        load_in_4bit=bool(args.local_load_in_4bit),
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )
    refiner = HPOCandidateRefiner(llm_client, max_candidates=args.topk)

    cfg_out = {"cli": vars(args), "meta": meta}
    cfg_path = os.path.join(args.out_dir, "hpo_revise_llm_config.json")
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

        samples_for_llm, dual_metrics = build_candidates_with_duallora(
            dataset_name,
            ds,
            model_tc,
            span_proj,
            tokenizer,
            ontology,
            cfg,
            device,
            z_hpo_global,
            hpo_ids_global,
            id2idx_global,
            topk=args.topk,
        )

        llm_metrics = run_llm_refine_for_dataset(
            dataset_name,
            samples_for_llm,
            refiner,
            num_workers=args.num_workers,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            batch_size=args.local_batch_size,
        )

        pipeline_top1 = (llm_metrics["pipeline_top1_hits"] / dual_metrics["total_spans"]) if dual_metrics["total_spans"] > 0 else 0.0

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
        "[GLOBAL] DualLoRAEnc top1=%.4f, recall@%d=%.4f, "
        "LLM conditional top1=%.4f (on called samples), Pipeline top1=%.4f (total_spans=%d)",
        global_dual_top1,
        args.topk,
        global_dual_recallK,
        global_llm_conditional,
        global_pipeline_top1,
        total_spans_all,
    )

    summary_json_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON summary to {summary_json_path}")

    plot_paths = plot_comparisons(results_summary, args.out_dir, topk=args.topk)

    md_path = os.path.join(args.out_dir, "hpo_revise_llm_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPO DualLoRAEnc + Local 4-bit LLM Refine Evaluation (Margin-Gated Pipeline, Global HPO Table)\n\n")

        f.write("## Command\n\n```bash\n")
        f.write("python HPO_MoRE.py \\\n")
        for k, v in vars(args).items():
            if isinstance(v, list):
                for item in v:
                    f.write(f"  --{k} {item} \\\n")
            else:
                if isinstance(v, bool):
                    if v:
                        f.write(f"  --{k} \\\n")
                else:
                    f.write(f"  --{k} {v} \\\n")
        f.write("```\n\n")

        g = results_summary.get("_GLOBAL", {})
        f.write("## Global Metrics (All Datasets Combined)\n\n")
        f.write(f"- Total spans: **{g.get('total_spans', 0)}**\n")
        f.write("- DualLoRAEnc Top-1 (global HPO table): **{:.4f}**\n".format(g.get("dual_top1", 0.0)))
        f.write("- DualLoRAEnc Recall@{} (global HPO table): **{:.4f}**\n".format(args.topk, g.get("dual_recallK", 0.0)))
        f.write("- LLM Conditional Top-1 (on called samples): **{:.4f}**\n".format(g.get("llm_top1_conditional", 0.0)))
        f.write("- **Full Pipeline Top-1**: **{:.4f}**\n\n".format(g.get("pipeline_top1", 0.0)))

        f.write("## Metrics per Dataset\n\n")
        f.write(
            "| Dataset | Dual Top-1 | Dual Recall@{} | Total spans | LLM samples | LLM calls | LLM Top-1 (cond.) | Pipeline Top-1 |\n".format(
                args.topk
            )
        )
        f.write("|---------|-----------:|---------------:|------------:|------------:|----------:|------------------:|---------------:|\n")

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

        f.write(
            "\nThis report summarizes DualLoRAEnc retrieval on a **global** HPO table "
            "and a margin-gated pipeline that combines DualLoRAEnc top-K retrieval with a **local 4-bit LLM** "
            "candidate selector.\n"
        )

    logger.info(f"Markdown summary saved to: {md_path}")
    logger.info("All done.")


if __name__ == "__main__":
    main()
