#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dump_duallora_rerank_examples.py  (STANDALONE, NO LLM)

What it does:
- Runs ONLY DualLoRAEnc retrieval against a GLOBAL HPO table (ALL ontology IDs)
- Builds top-K candidate lists for spans
- Dumps N examples into ONE JSON file for your next-step reranker testing

Saved JSON includes:
- context_window (window)
- mention text + span offsets
- gold_id
- dual_topk candidates (each includes full enriched def: DEF + LLM_DEF + ADD_DEF)
- dual scores and margin (top1 - top2)

Output dir example:
  /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/test_data

Usage:
python /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/example.py \
  --eval_root /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GeneReviews \
  --hpo_json /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/hpo_data/hpo_enriched_with_llm.json \
  --model_dir /cluster/home/gw/Backend_project/NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --init_encoder_from /cluster/home/gw/Backend_project/NER/tuned/intention \
  --span_ckpt /cluster/home/gw/Backend_project/NER/tuned/duallora_span_llm/hpoid_span_best.pt \
  --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/HPO_MoRE/test_data \
  --topk 35 \
  --n_examples 200 \
  --policy gold_in_topk \
  --pick random \
  --seed 42 \
  --hpo_chunk_size 512 \
  --max_context_chars 512

Policies:
- gold_in_topk : only dump examples where gold ∈ topK (good for measuring reranker improvements)
- all          : dump any examples (gold may be missing)

Pick:
- random / first / min_margin (hard cases)

Notes:
- Requires your project imports:
  train_hpoid_span_contrastive.py provides:
    HPOConfig, HPOOntology, TokenCRFWrapper, HPOIDSpanPairDataset, SpanProj,
    encode_spans, encode_hpo_gold_table, load_ner_tc_and_tokenizer
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional

import torch
from tqdm import tqdm

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

logger = logging.getLogger("DUMP_DUALLORA_EXAMPLES")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")

SPAN = Tuple[int, int]


def get_hpo_prompt_info(ontology: HPOOntology, hpo_id: str) -> Dict[str, Any]:
    """
    Build enriched candidate payload from ontology record:
      - Name / Synonym / Def
      - llm_def / llm_add_def
    """
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

    # unified "def_text" used by downstream rerankers
    lines = [f"[HPO_ID] {hid}", f"[NAME] {name}"]
    if syns:
        lines.append(f"[SYN] {'; '.join(syns)}")
    if orig_def:
        lines.append(f"[DEF] {orig_def}")
    if llm_def and llm_def != orig_def:
        lines.append(f"[LLM_DEF] {llm_def}")
    if llm_add_def:
        lines.append(f"[ADD_DEF] {llm_add_def}")
    def_text = "\n".join(lines)

    return {
        "hpo_id": hid,
        "hpo_name": name,
        "hpo_synonyms": syns,
        "hpo_def_text": def_text,
        "hpo_orig_def": orig_def,
        "hpo_llm_def": llm_def,
        "hpo_add_def": llm_add_def,
    }


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
    if not all_hpo_ids:
        raise RuntimeError("Ontology has no HPO IDs.")

    z_chunks: List[torch.Tensor] = []
    hpo_ids_full: List[str] = []

    logger.info(f"[GlobalHPO] Encoding ALL HPO IDs: N={len(all_hpo_ids)} chunk={chunk_size}")
    for start in tqdm(range(0, len(all_hpo_ids), chunk_size), desc="[GlobalHPO]", leave=False):
        chunk_ids = all_hpo_ids[start : start + chunk_size]
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
        raise RuntimeError("Failed to build global HPO table (no valid chunks).")

    z_hpo = torch.cat(z_chunks, dim=0)
    id2idx = {hid: i for i, hid in enumerate(hpo_ids_full)}
    logger.info(f"[GlobalHPO] Done: table={tuple(z_hpo.shape)} on {z_hpo.device}")
    return z_hpo, hpo_ids_full, id2idx


def iter_duallora_topk(
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
    topk: int,
    max_context_chars: int,
):
    """
    Yield per-span result dict with topK candidates and scores.
    """
    B = max(1, min(cfg.batch_size, 32))
    global_idx = 0

    for i in tqdm(range(0, len(ds), B), desc="[DualLoRAEnc] spans", leave=False):
        chunk = [ds[j] for j in range(i, min(i + B, len(ds)))]
        if not chunk:
            continue

        left_texts = [ex["left_text"] for ex in chunk]
        left_spans = [ex["left_span"] for ex in chunk]
        gold_ids = [ex["hpo_id"] for ex in chunk]

        z_left = encode_spans(model_tc, span_proj, tokenizer, left_texts, left_spans, device, cfg.max_len)
        sims = z_left @ z_hpo.t()
        inner_topk = min(topk, sims.size(1))
        vals, idxs = torch.topk(sims, k=inner_topk, dim=-1)

        vals = vals.detach().cpu()
        idxs = idxs.detach().cpu()

        for row, ex in enumerate(chunk):
            left_text = ex["left_text"]
            c0, c1 = ex["left_span"]
            c0 = max(0, min(c0, len(left_text)))
            c1 = max(0, min(c1, len(left_text)))

            mention = left_text[c0:c1]
            context = left_text[:max_context_chars] if len(left_text) > max_context_chars else left_text
            gold_id = gold_ids[row]
            gold_idx = id2idx.get(gold_id, None)

            preds = idxs[row].tolist()
            gold_in_topk = (gold_idx is not None and gold_idx in preds)
            dual_top1 = False
            if gold_idx is not None and preds:
                dual_top1 = (preds[0] == gold_idx)

            # build candidates payload
            cand_list = []
            for r in range(inner_topk):
                hpo_row = int(idxs[row, r].item())
                score = float(vals[row, r].item())
                hid = hpo_ids_vec[hpo_row]
                info = get_hpo_prompt_info(ontology, hid)
                info["score"] = score
                info["rank"] = r
                cand_list.append(info)

            if cand_list:
                best_score = cand_list[0]["score"]
                second_score = cand_list[1]["score"] if len(cand_list) > 1 else best_score
                margin = float(best_score - second_score)
                dual_best_id = cand_list[0]["hpo_id"]
            else:
                margin = 0.0
                dual_best_id = None

            yield {
                "idx": global_idx,
                "context_window": context,
                "mention": mention,
                "mention_span": [int(c0), int(c1)],
                "gold_id": gold_id,
                "gold_in_topk": bool(gold_in_topk),
                "dual_top1": bool(dual_top1),
                "dual_best_id": dual_best_id,
                "dual_margin": margin,
                "dual_topk": cand_list,
            }
            global_idx += 1


def select_examples(
    stream,
    n_examples: int,
    policy: str,
    pick: str,
    seed: int,
):
    """
    policy: "gold_in_topk" or "all"
    pick: "first" | "random" | "min_margin"
    """
    buf: List[Dict[str, Any]] = []

    def _accept(x: Dict[str, Any]) -> bool:
        if policy == "gold_in_topk":
            return bool(x.get("gold_in_topk", False))
        return True

    if pick == "first":
        for x in stream:
            if _accept(x):
                buf.append(x)
                if len(buf) >= n_examples:
                    break
        return buf

    if pick == "min_margin":
        # collect a working set then take smallest margins
        for x in stream:
            if _accept(x):
                buf.append(x)
                if len(buf) >= max(n_examples * 5, n_examples):
                    break
        buf.sort(key=lambda d: float(d.get("dual_margin", 1e9)))
        return buf[:n_examples]

    # random reservoir sampling
    rng = torch.Generator().manual_seed(seed)
    seen = 0
    for x in stream:
        if not _accept(x):
            continue
        seen += 1
        if len(buf) < n_examples:
            buf.append(x)
            continue
        j = int(torch.randint(low=0, high=seen, size=(1,), generator=rng).item())
        if j < n_examples:
            buf[j] = x
    return buf


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=str, required=True)
    ap.add_argument("--hpo_json", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, required=True)
    ap.add_argument("--init_encoder_from", type=str, default=None)
    ap.add_argument("--span_ckpt", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default=None)

    ap.add_argument("--topk", type=int, default=35)
    ap.add_argument("--n_examples", type=int, default=200)
    ap.add_argument("--policy", type=str, default="gold_in_topk", choices=["gold_in_topk", "all"])
    ap.add_argument("--pick", type=str, default="random", choices=["first", "random", "min_margin"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--hpo_chunk_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_context_chars", type=int, default=512)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = os.path.basename(args.eval_root.rstrip("/"))
    logger.info(f"Device={device} dataset={dataset_name}")

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
    logger.info(f"Loaded ontology: {len(ontology.data)} nodes")

    tokenizer, model_tc, meta = load_ner_tc_and_tokenizer(
        args.backbone,
        args.init_encoder_from,
        args.model_dir,
        cfg,
    )
    model_tc.to(device).eval()

    # span head
    if not os.path.isfile(args.span_ckpt):
        raise FileNotFoundError(args.span_ckpt)
    ckpt = torch.load(args.span_ckpt, map_location="cpu")
    span_dim = ckpt.get("cfg", {}).get("hpoid_dim", 256)
    hidden_size = model_tc.base.config.hidden_size
    span_proj = SpanProj(in_dim=hidden_size, out_dim=span_dim, dropout=0.0).to(device)
    span_proj.load_state_dict(ckpt["span_proj_state"])
    span_proj.eval()
    logger.info(f"Loaded span head: dim={span_dim}")

    # global HPO table
    z_hpo, hpo_ids_vec, id2idx = build_global_hpo_table(
        model_tc=model_tc,
        span_proj=span_proj,
        tokenizer=tokenizer,
        ontology=ontology,
        cfg=cfg,
        device=device,
        chunk_size=args.hpo_chunk_size,
    )

    # dataset
    ds = HPOIDSpanPairDataset(
        roots=[args.eval_root],
        ontology=ontology,
        max_context_chars=256,  # dataset loader internal; we still output our own max_context_chars below
        max_syn=3,
    )
    logger.info(f"Dataset spans={len(ds)}")

    stream = iter_duallora_topk(
        ds=ds,
        model_tc=model_tc,
        span_proj=span_proj,
        tokenizer=tokenizer,
        ontology=ontology,
        cfg=cfg,
        device=device,
        z_hpo=z_hpo,
        hpo_ids_vec=hpo_ids_vec,
        id2idx=id2idx,
        topk=args.topk,
        max_context_chars=args.max_context_chars,
    )

    examples = select_examples(
        stream=stream,
        n_examples=args.n_examples,
        policy=args.policy,
        pick=args.pick,
        seed=args.seed,
    )

    # quick stats
    n = len(examples)
    gold_in = sum(1 for x in examples if x.get("gold_in_topk"))
    top1 = sum(1 for x in examples if x.get("dual_top1"))
    logger.info(f"Selected examples={n} gold_in_topk={gold_in}/{n} dual_top1={top1}/{n}")

    payload = {
        "meta": {
            "dataset": dataset_name,
            "eval_root": args.eval_root,
            "hpo_json": args.hpo_json,
            "backbone": args.backbone,
            "model_dir": args.model_dir,
            "init_encoder_from": args.init_encoder_from,
            "span_ckpt": args.span_ckpt,
            "topk": args.topk,
            "n_examples": args.n_examples,
            "policy": args.policy,
            "pick": args.pick,
            "seed": args.seed,
            "hpo_table_size": int(z_hpo.size(0)),
            "span_dim": int(z_hpo.size(1)),
        },
        "examples": examples,
    }

    if args.out_name:
        out_name = args.out_name
    else:
        out_name = f"duallora_rerank_examples_{dataset_name}_top{args.topk}_{args.policy}_{args.pick}.json"

    out_path = os.path.join(args.out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"[DUMP] Wrote => {out_path}")


if __name__ == "__main__":
    main()
