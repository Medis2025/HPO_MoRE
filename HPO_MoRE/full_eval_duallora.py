#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation for HPO span-contrastive encoder:

- 对每个数据集单独验证：
  * DualLoRAEnc only: span encoder 全库检索（ALL-HPO FULL）
  * FUZZY only: Stage-A (EXACT+FUZZY) + 层级剪枝 + span encoder + 层级重排（PIPELINE-FUZZY）
  * PIPELINE_all: Stage-A (lexical + fuzzy + direct_NER_bert) + 层级剪枝 + span encoder + 层级重排（A+新B）

- 使用 train_hpoid_span_contrastive.py 中的组件：
  HPOConfig, HPOOntology, TokenCRFWrapper,
  HPOIDSpanPairDataset, SpanProj, encode_spans, encode_hpo_gold_table, load_ner_tc_and_tokenizer

Example:

python eval_hpoid_span_contrastive.py \
  --eval_roots \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GeneReviews \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GSC+ \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/ID-68 \
  --hpo_json /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/hpo.json \
  --model_dir /cluster/home/gw/Backend_project/NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --init_encoder_from /cluster/home/gw/Backend_project/NER/tuned/intention \
  --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/VAL \
  --ckpt_path /cluster/home/gw/Backend_project/NER/tuned/hpoid_span_contrastive/hpoid_span_best.pt \
  --batch_size 32 \
  --max_len 512 \
  --stageA_topk 15
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
#                      "expandable_segments:False,garbage_collection_threshold:0.9,max_split_size_mb:128")
import json
import time
import logging
import argparse
from typing import List, Dict, Tuple, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---- optional deps for resource / plotting ----
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---- import from your training script ----
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

logger = logging.getLogger("HPOIDSpanEval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
)

SPAN = Tuple[int, int]


# =============================================================================
# Small helpers: resource measurement
# =============================================================================
def get_cpu_mem_mb() -> float:
    if not HAS_PSUTIL:
        return -1.0
    try:
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 ** 2)
    except Exception:
        return -1.0


def reset_gpu_peak(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass


def get_gpu_peak_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return -1.0
    try:
        return float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
    except Exception:
        return -1.0


# =============================================================================
# Plotting class
# =============================================================================
class EvalPlotter:
    """
    负责把多数据集、多方法的指标画成漂亮一点的图：
      - Top1 / Top5 / Time / GPU / CPU
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir

    def _plot_metric(
        self,
        results_summary: Dict[str, Dict[str, Dict[str, Any]]],
        metric: str,
        ylabel: str,
        filename: str,
        methods: List[str],
        is_ratio: bool = False,
    ) -> str:
        if not HAS_MPL:
            logger.warning(f"[PLOT] matplotlib not available, skip plotting {metric}.")
            return ""

        datasets = sorted(results_summary.keys())
        if not datasets:
            return ""

        # 准备数据：每个 method 一个数组
        values_by_method: Dict[str, List[float]] = {m: [] for m in methods}
        for ds in datasets:
            m_dict = results_summary[ds]
            for m in methods:
                v = m_dict.get(m, {}).get(metric, None)
                if v is None:
                    values_by_method[m].append(0.0)
                else:
                    values_by_method[m].append(float(v))

        plt.style.use("ggplot")
        fig_width = max(6.0, 1.5 * len(datasets))
        plt.figure(figsize=(fig_width, 4.5))

        x = torch.arange(len(datasets)).tolist()
        total_methods = len(methods)
        bar_width = 0.8 / max(1, total_methods)

        for idx_m, m in enumerate(methods):
            xs = [xi + (idx_m - total_methods / 2) * bar_width + bar_width / 2 for xi in x]
            plt.bar(xs, values_by_method[m], width=bar_width, label=m)

        plt.xticks(x, datasets, rotation=30, ha="right")
        plt.ylabel(ylabel)
        if is_ratio:
            plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.out_dir, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()
        logger.info(f"[PLOT] Saved {metric} plot to {out_path}")
        return out_path

    def plot_all(self, results_summary: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, str]:
        methods = ["DualLoRAEnc", "FUZZY_ONLY", "PIPELINE_all"]
        paths = {}
        paths["top1"] = self._plot_metric(
            results_summary, "top1", "Top-1 Accuracy", "perf_top1.png", methods, is_ratio=True
        )
        paths["top5"] = self._plot_metric(
            results_summary, "top5", "Top-5 Accuracy", "perf_top5.png", methods, is_ratio=True
        )
        paths["time"] = self._plot_metric(
            results_summary, "time_sec", "Time (s)", "perf_time.png", methods, is_ratio=False
        )
        paths["gpu"] = self._plot_metric(
            results_summary, "gpu_mem_mb", "Peak GPU Memory (MB)", "perf_gpu_mem.png", methods, is_ratio=False
        )
        paths["cpu"] = self._plot_metric(
            results_summary, "cpu_mem_mb", "CPU Memory (MB)", "perf_cpu_mem.png", methods, is_ratio=False
        )
        return paths


# =============================================================================
# Hierarchy helper (StageA prune + StageB re-ranking)
# =============================================================================
class HPOHierarchyHelper:
    """
    统一处理 HPO 层级相关逻辑：
      * prune_candidates: 在 StageA 候选上做层级剪枝（保留更具体的叶子）
      * rerank_indices: 在 StageB 的 sim 基础上做层级一致性重排
    """

    def __init__(
        self,
        ontology: HPOOntology,
        depth_margin: float = 0.05,
        child_delta: float = 0.02,
        child_penalty: float = 0.05,
        depth_lambda: float = 0.10,
    ):
        self.onto = ontology
        self.depth_margin = float(depth_margin)
        self.child_delta = float(child_delta)
        self.child_penalty = float(child_penalty)
        self.depth_lambda = float(depth_lambda)

        if hasattr(self.onto, "depth") and self.onto.depth:
            self.max_depth = max(self.onto.depth.values())
        else:
            self.max_depth = 1

    def _depth_norm(self, hid: str) -> float:
        d = 0
        try:
            d = self.onto.get_depth(hid)
        except Exception:
            d = 0
        if self.max_depth <= 0:
            return 0.0
        return float(d) / float(self.max_depth)

    # ---------- StageA: hierarchical prune ----------
    def prune_candidates(
        self,
        scored: List[Tuple[str, float]],
        max_k: int,
    ) -> List[str]:
        """
        输入：[(hid, score)]，score 已经是 StageA 合成分数
        策略：
          1. 先按 (score, depth) 降序排序
          2. 遍历排序后的列表，对每个 (h,s)：
             - 如果已经保留的某个 hk 是 h 的后代，且 sk >= s - depth_margin，则丢掉 h
             - 否则保留
        返回：剪枝后的 HPO Id 列表（按保留顺序，截断到 max_k）
        """
        if not scored:
            return []

        # 标准化 Id（Alt_id -> 主 Id），并记录原始 score
        norm_scored: List[Tuple[str, float]] = []
        for hid, s in scored:
            try:
                h_norm = self.onto.resolve_id(hid)
            except Exception:
                h_norm = hid
            norm_scored.append((h_norm, float(s)))

        # 按分数 + 深度排序：score 高优先，其次 depth 深优先
        norm_scored.sort(
            key=lambda x: (x[1], self._depth_norm(x[0])),
            reverse=True,
        )

        kept: List[Tuple[str, float]] = []
        for hid, s in norm_scored:
            drop = False
            for hk, sk in kept:
                # 如果 hk 是 hid 的后代，且 sk 不比 s 差太多，则丢掉 hid
                try:
                    if self.onto.is_ancestor(hid, hk) and sk >= s - self.depth_margin:
                        drop = True
                        break
                except Exception:
                    # 任何异常都不剪
                    continue
            if not drop:
                kept.append((hid, s))
            if len(kept) >= max_k:
                break

        # 去重并截断
        out: List[str] = []
        seen: set = set()
        for h, _ in kept:
            if h in seen:
                continue
            seen.add(h)
            out.append(h)
            if len(out) >= max_k:
                break
        return out

    # ---------- StageB: hierarchical consistency re-ranking ----------
    def rerank_indices(
        self,
        cand_indices: List[int],
        sims_tensor: torch.Tensor,
        hpo_ids_vec: List[str],
    ) -> List[int]:
        """
        输入：
          cand_indices: 在 z_hpo 里的下标列表
          sims_tensor:  StageB 计算出的相似度 [K]（和 cand_indices 对齐）
          hpo_ids_vec:  z_hpo 的 HPO Id 顺序，用于取出 hid

        步骤：
          1. base_score = sims
          2. 加 depth bias： + depth_lambda * depth_norm(h)
          3. 层级一致性校正：
             - 若存在某个同一候选集中的 child，使得:
                 is_ancestor(h, child) == True 且 score(child) >= score(h) + child_delta
               则对 h 施加 child_penalty
          4. 按 final_score 降序排序，返回新的 cand_indices 顺序
        """
        if not cand_indices or sims_tensor.numel() == 0:
            return cand_indices

        sims = sims_tensor.detach().cpu().tolist()
        if len(sims) != len(cand_indices):
            # 防御：尺寸不一致则放弃层级 re-rank
            return cand_indices

        # 1. base scores
        base_scores: Dict[int, float] = {
            idx: float(score) for idx, score in zip(cand_indices, sims)
        }

        # 2. depth bias
        depth_bias: Dict[int, float] = {}
        for idx in cand_indices:
            hid = hpo_ids_vec[idx]
            depth_bias[idx] = self.depth_lambda * self._depth_norm(hid)

        # 3. child consistency penalty
        penalty: Dict[int, float] = {idx: 0.0 for idx in cand_indices}
        # 预取 hid 列表
        cand_hids: Dict[int, str] = {idx: hpo_ids_vec[idx] for idx in cand_indices}

        for idx_a in cand_indices:
            h_a = cand_hids[idx_a]
            s_a = base_scores[idx_a] + depth_bias.get(idx_a, 0.0)
            # 看看是否有更具体的 child 表现明显更好
            for idx_b in cand_indices:
                if idx_a == idx_b:
                    continue
                h_b = cand_hids[idx_b]
                s_b = base_scores[idx_b] + depth_bias.get(idx_b, 0.0)
                if s_b < s_a + self.child_delta:
                    continue
                try:
                    if self.onto.is_ancestor(h_a, h_b):
                        # h_a 是祖先，child 表现明显更好，给祖先一点惩罚
                        penalty[idx_a] += self.child_penalty
                        break
                except Exception:
                    continue

        # 4. 合成最终分数并排序
        final_scores: List[Tuple[int, float]] = []
        for idx in cand_indices:
            fs = base_scores[idx] + depth_bias.get(idx, 0.0) - penalty.get(idx, 0.0)
            final_scores.append((idx, fs))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        new_order = [idx for idx, _ in final_scores]
        return new_order


# =============================================================================
# Stage-A: lexical + fuzzy + direct_NER_bert candidate generator (top-K) with modes
# =============================================================================
def ensure_fuzzy_cache(ontology: HPOOntology):
    """
    Build a NER_FUZZY key cache on ontology if not exists:
    - ontology._keys: list of normalized term keys (ontology.term2ids.keys())
    对齐 CandidateFinder.find_ranked_for_eval 里的 NER_FUZZY 逻辑：
      * 在 term2ids 的 key 空间做 NER_FUZZY
    """
    if hasattr(ontology, "_keys"):
        return
    keys = list(getattr(ontology, "term2ids", {}).keys())
    ontology._keys = keys
    logger.info(f"[StageA] Built NER_FUZZY key cache with {len(keys)} keys from term2ids.")


def stageA_ranked(
    mention_text: str,
    ontology: HPOOntology,
    fuzzy_limit: int = 256,
    z_left_row: torch.Tensor = None,
    z_hpo: torch.Tensor = None,
    hpo_ids_vec: List[str] = None,
    mode: str = "all",  # "all" | "ner_only" | "fuzzy_only"
) -> List[Tuple[str, float]]:
    """
    返回带分数的 Stage-A 候选：
      [(hid, stageA_score), ...]  未截断 / 未剪枝

    mode 说明：
      - "all": EXACT + NER_FUZZY + direct_NER_bert (原逻辑)
      - "ner_only": 只用 direct_NER_bert（不使用 EXACT/FUZZY）
      - "fuzzy_only": 只用 EXACT + FUZZY（不使用 direct_NER_bert）

    stageA_score(hid) = max { EXACT_score, FUZZY_score, direct_NER_bert_score }
    """
    mention_text = (mention_text or "").strip()
    if not mention_text:
        return []

    mode = (mode or "all").lower()

    # ---------- EXACT ----------
    exact_map: Dict[str, float] = {}
    if mode != "ner_only":  # ner_only 不使用 lexical
        try:
            norm_key = ontology._norm(mention_text)
            for hid in ontology.term2ids.get(norm_key, set()):
                exact_map[hid] = 1.0
        except Exception:
            exact_map = {}

    # ---------- FUZZY（对 term2ids.key） ----------
    fuzzy_map: Dict[str, float] = {}
    if mode != "ner_only":
        try:
            ensure_fuzzy_cache(ontology)
            from rapidfuzz import process, fuzz

            if hasattr(ontology, "_norm_adv"):
                key = ontology._norm_adv(mention_text)
            else:
                key = ontology._norm(mention_text)

            hits = process.extract(
                key,
                ontology._keys,
                scorer=fuzz.token_set_ratio,
                limit=min(fuzzy_limit, max(1, len(ontology._keys))),
            )
            thr = 88.0
            for cand, rf_score, _ in hits:
                if rf_score < thr:
                    continue
                for hid in ontology.term2ids.get(cand, []):
                    s = (max(rf_score, 0.0) / 100.0) * 0.5
                    prev = fuzzy_map.get(hid, 0.0)
                    if s > prev:
                        fuzzy_map[hid] = s
        except Exception:
            fuzzy_map = {}

    # ---------- direct_NER_bert ----------
    direct_map: Dict[str, float] = {}
    if mode != "fuzzy_only":
        if (
            z_left_row is not None
            and isinstance(z_left_row, torch.Tensor)
            and z_hpo is not None
            and isinstance(z_hpo, torch.Tensor)
            and hpo_ids_vec is not None
        ):
            try:
                if z_left_row.dim() == 1:
                    sims = torch.mv(z_hpo, z_left_row)  # [N]
                else:
                    sims = torch.matmul(z_hpo, z_left_row.squeeze(0))  # [N]
                sims_list = sims.detach().cpu().tolist()
                for idx, hid in enumerate(hpo_ids_vec):
                    direct_map[hid] = float(sims_list[idx])
            except Exception:
                direct_map = {}

    # ---------- 汇总 ----------
    all_ids = set(exact_map.keys()) | set(fuzzy_map.keys()) | set(direct_map.keys())
    scored: List[Tuple[str, float]] = []
    for hid in all_ids:
        s_exact = exact_map.get(hid, None)
        s_fuzzy = fuzzy_map.get(hid, None)
        s_direct = direct_map.get(hid, None)
        scores = [x for x in (s_exact, s_fuzzy, s_direct) if x is not None]
        if not scores:
            continue
        combined = max(scores)
        scored.append((hid, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def stageA_candidates(
    mention_text: str,
    ontology: HPOOntology,
    topk: int = 15,
    fuzzy_limit: int = 256,
    z_left_row: torch.Tensor = None,
    z_hpo: torch.Tensor = None,
    hpo_ids_vec: List[str] = None,
) -> List[str]:
    """
    兼容原有接口：只是简单调用 stageA_ranked(mode="all")，然后截断 topk 且不做层级剪枝。
    在 PIPELINE_all 里我们会走 ranked + HPOHierarchyHelper.prune_candidates。
    """
    ranked = stageA_ranked(
        mention_text=mention_text,
        ontology=ontology,
        fuzzy_limit=fuzzy_limit,
        z_left_row=z_left_row,
        z_hpo=z_hpo,
        hpo_ids_vec=hpo_ids_vec,
        mode="all",
    )
    return [hid for hid, _ in ranked[:topk]]


# =============================================================================
# Generic evaluation helpers
# =============================================================================
@torch.no_grad()
def eval_full_retrieval(
    dataset_name: str,
    ds: HPOIDSpanPairDataset,
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer,
    ontology: HPOOntology,
    cfg: HPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    """
    DualLoRAEnc only：span encoder 对比 HPO 子库（ALL-HPO FULL：仅当前 dataset 出现过的 HPO）:
    - 对 ds 中所有 mention spans encode 成 z_span
    - 对 HPO 子集（本 dataset 实际出现过的 gold HPO）构建 gold table z_hpo
    - top1 / top5 准确率
    - precision/recall/F1: 采用 per-mention 统计方法（基于 top-5 命中）
      * 若 N 个 mention 中，有 H 个 mention 的 gold ∈ top-5，
        则 P = R = F1 = H / N
    """
    logger.info(f"[DualLoRAEnc] Evaluating dataset={dataset_name} with ALL-HPO retrieval...")

    # 只用当前 dataset 出现过的 HPO IDs 构建表
    hpo_ids_table = sorted({ex["hpo_id"] for ex in ds if ex["hpo_id"] in ontology.data})
    logger.info(f"[DualLoRAEnc] dataset={dataset_name} has {len(hpo_ids_table)} unique HPO IDs.")

    z_hpo, hpo_ids_vec = encode_hpo_gold_table(
        model_tc,
        span_proj,
        tokenizer,
        ontology,
        hpo_ids_table,
        device=device,
        max_len=cfg.max_len,
    )
    if z_hpo.numel() == 0:
        logger.warning("[DualLoRAEnc] Empty HPO embedding table.")
        return {"top1": 0.0, "top5": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    id2idx = {hid: i for i, hid in enumerate(hpo_ids_vec)}
    logger.info(f"[DualLoRAEnc] HPO table size (DATASET-SUBSET): {len(id2idx)}")

    # 遍历 dataset spans
    BATCH = cfg.batch_size
    top1_hits = 0
    top5_hits = 0
    total = 0

    for i in tqdm(range(0, len(ds), BATCH), desc=f"[DualLoRAEnc] {dataset_name}", leave=False):
        chunk = [ds[j] for j in range(i, min(i + BATCH, len(ds)))]
        left_texts = [ex["left_text"] for ex in chunk]
        left_spans = [ex["left_span"] for ex in chunk]
        hids_gold = [ex["hpo_id"] for ex in chunk]

        z_left = encode_spans(
            model_tc, span_proj, tokenizer, left_texts, left_spans, device, cfg.max_len
        )  # [b, D]

        sims = z_left @ z_hpo.t()  # [b, N]
        _, topi = torch.topk(sims, k=min(5, sims.size(1)), dim=-1)

        for row, hid_true in enumerate(hids_gold):
            total += 1
            if hid_true not in id2idx:
                continue
            true_idx = id2idx[hid_true]
            preds = topi[row].tolist()
            if preds and preds[0] == true_idx:
                top1_hits += 1
            if true_idx in preds:
                top5_hits += 1

    top1 = top1_hits / max(1, total)
    top5 = top5_hits / max(1, total)

    # per-mention precision/recall/F1（基于 top-5 命中）
    precision = top5
    recall = top5
    f1 = top5

    logger.info(
        f"[DualLoRAEnc] dataset={dataset_name} HPO-ID top1={top1:.4f}, top5={top5:.4f} "
        f"(total={total}), P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
    )
    return {
        "top1": top1,
        "top5": top5,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


@torch.no_grad()
def eval_pipeline_AplusB(
    dataset_name: str,
    ds: HPOIDSpanPairDataset,
    model_tc: TokenCRFWrapper,
    span_proj: SpanProj,
    tokenizer,
    ontology: HPOOntology,
    cfg: HPOConfig,
    device: torch.device,
    stageA_topk: int = 15,
    stageA_mode: str = "all",   # "all" | "fuzzy_only" | "ner_only" (这里用 "all" / "fuzzy_only")
    mode_tag: str = "PIPELINE_all", # 用于 log 区分 PIPELINE_all / FUZZY_ONLY
) -> Dict[str, float]:
    """
    PIPELINE_all 模式：A+新B（带层级）

    Stage A:
      - 根据 stageA_mode:
          * "all": lexical + NER_FUZZY + direct_NER_bert (原 A+新B)
          * "fuzzy_only": EXACT+NER_FUZZY (不含 direct_NER_bert)
      - StageA_ranked 得到 [(hid, score)]，再用 HPOHierarchyHelper 做 hierarchical prune
      - 召回 top-K HPO 候选
    Stage B:
      - span encoder + HPO embedding 表
      - 先算 sims，然后用 HPOHierarchyHelper.rerank_indices 做层级一致性重排
      - 最终取 top1 / top5

    输出：
      - recallA: gold 是否出现在 Stage A 的 top-K 候选中（剪枝后）
      - top1/top5: 在 PIPELINE_all 模式下的 HPO-ID top-1 / top-5 准确率
      - precision/recall/f1: per-mention 指标（基于 pipeline top-5 命中）
    """
    logger.info(
        f"[{mode_tag}] Evaluating dataset={dataset_name} with "
        f"StageA(mode={stageA_mode}, top-{stageA_topk}) + hierarchy + span encoder..."
    )

    # 同样只在当前 dataset 的 HPO 子集上建 gold 表
    hpo_ids_table = sorted({ex["hpo_id"] for ex in ds if ex["hpo_id"] in ontology.data})
    logger.info(f"[{mode_tag}] dataset={dataset_name} has {len(hpo_ids_table)} unique HPO IDs.")

    z_hpo, hpo_ids_vec = encode_hpo_gold_table(
        model_tc,
        span_proj,
        tokenizer,
        ontology,
        hpo_ids_table,
        device=device,
        max_len=cfg.max_len,
    )
    if z_hpo.numel() == 0:
        logger.warning(f"[{mode_tag}] Empty HPO embedding table.")
        return {
            "recallA": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    id2idx = {hid: i for i, hid in enumerate(hpo_ids_vec)}
    logger.info(f"[{mode_tag}] HPO table size (DATASET-SUBSET): {len(id2idx)}")

    # 层级 helper
    hier = HPOHierarchyHelper(ontology)

    BATCH = cfg.batch_size

    recallA_hits = 0
    top1_hits = 0
    top5_hits = 0
    total = 0

    for i in tqdm(
        range(0, len(ds), BATCH),
        desc=f"[{mode_tag}] {dataset_name}",
        leave=False,
    ):
        chunk = [ds[j] for j in range(i, min(i + BATCH, len(ds)))]
        left_texts = [ex["left_text"] for ex in chunk]
        left_spans = [ex["left_span"] for ex in chunk]
        hids_gold = [ex["hpo_id"] for ex in chunk]

        # mention text = span substring from left_text
        mentions = []
        for ex in chunk:
            lt = ex["left_text"]
            c0, c1 = ex["left_span"]
            c0 = max(0, min(c0, len(lt)))
            c1 = max(0, min(c1, len(lt)))
            m = lt[c0:c1]
            mentions.append(m)

        # Stage B 先算 z_left
        z_left = encode_spans(
            model_tc, span_proj, tokenizer, left_texts, left_spans, device, cfg.max_len
        )  # [b, D]

        # 逐个样本应用 Stage A + 层级剪枝 + StageB 层级重排
        for row, (mention, hid_true) in enumerate(zip(mentions, hids_gold)):
            total += 1

            # Stage A ranked
            ranked = stageA_ranked(
                mention_text=mention,
                ontology=ontology,
                fuzzy_limit=256,
                z_left_row=z_left[row],
                z_hpo=z_hpo,
                hpo_ids_vec=hpo_ids_vec,
                mode=stageA_mode,
            )

            if not ranked:
                # Stage-A 完全失败：pipeline 也视为没命中
                continue

            # 层级剪枝 -> cands
            cands = hier.prune_candidates(ranked, max_k=stageA_topk)

            if hid_true in cands:
                recallA_hits += 1

            if not cands:
                # Stage A 完全没召回任何候选，pipeline 视为失败
                continue

            # map candidate IDs -> indices in z_hpo
            cand_indices = [id2idx[h] for h in cands if h in id2idx]
            if not cand_indices:
                continue

            sims = z_left[row : row + 1, :] @ z_hpo[cand_indices, :].t()  # [1, K]
            sims = sims.squeeze(0)  # [K]

            # StageB 层级一致性 re-ranking
            new_order = hier.rerank_indices(
                cand_indices=cand_indices,
                sims_tensor=sims,
                hpo_ids_vec=hpo_ids_vec,
            )

            # 取 top-5 的全局 index
            if not new_order:
                continue
            top_indices = new_order[: min(5, len(new_order))]

            if hid_true in id2idx:
                true_idx = id2idx[hid_true]
                if top_indices and top_indices[0] == true_idx:
                    top1_hits += 1
                if true_idx in top_indices:
                    top5_hits += 1

    recallA = recallA_hits / max(1, total)
    top1 = top1_hits / max(1, total)
    top5 = top5_hits / max(1, total)

    # per-mention precision/recall/F1（基于 pipeline top-5 命中）
    precision = top5
    recall = top5
    f1 = top5

    logger.info(
        f"[{mode_tag}] dataset={dataset_name} "
        f"recallA@{stageA_topk}={recallA:.4f}, top1={top1:.4f}, top5={top5:.4f} "
        f"(total={total}), P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
    )
    return {
        "recallA": recallA,
        "top1": top1,
        "top5": top5,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# CLI & main
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Eval HPO span-contrastive encoder: DualLoRAEnc vs NER_FUZZY vs PIPELINE_all (with hierarchy)"
    )
    ap.add_argument(
        "--eval_roots",
        type=str,
        nargs="+",
        required=True,
        help="Eval roots (each with ann/ and corpus/). Will eval each root separately.",
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
        help="NER LoRA model dir (TokenCRFWrapper / PeftModel)",
    )
    ap.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="HF backbone path (e.g., BioLinkBERT-base)",
    )
    ap.add_argument(
        "--init_encoder_from",
        type=str,
        default=None,
        help="Optional encoder init checkpoint (e.g., intention NER ckpt)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Span-contrastive training out_dir (for config + tb + ckpt). "
            "Eval logs, new tb, plots and markdown summary will also be stored here."
        ),
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to span-projection checkpoint (default: <out_dir>/hpoid_span_best.pt)",
    )
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stageA_topk", type=int, default=15)
    return ap.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eval_tb_dir = os.path.join(args.out_dir, "eval_tb")
    os.makedirs(eval_tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=eval_tb_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load span-contrastive config (mainly to get hidden dim / etc.)
    cfg_path = os.path.join(args.out_dir, "hpoid_span_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_blob = json.load(f)
        hpo_cfg_dict = cfg_blob.get("hpo_cfg", {})
        # reconstruct HPOConfig minimally
        cfg = HPOConfig(
            backbone=args.backbone,
            init_encoder_from=args.init_encoder_from,
            model_dir=args.model_dir,
            hpo_json=args.hpo_json,
            max_len=args.max_len,
            batch_size=args.batch_size,
            stride=0,
            hpo_topk=5,
        )
        # attach temperature if present
        temp = cfg_blob.get("hpo_cfg", {}).get("hpoid_temp", None)
        if temp is not None:
            setattr(cfg, "hpoid_temp", float(temp))
    else:
        logger.warning(f"Config file not found: {cfg_path}, using basic HPOConfig.")
        cfg = HPOConfig(
            backbone=args.backbone,
            init_encoder_from=args.init_encoder_from,
            model_dir=args.model_dir,
            hpo_json=args.hpo_json,
            max_len=args.max_len,
            batch_size=args.batch_size,
            stride=0,
            hpo_topk=5,
        )

    # Ontology
    ontology = HPOOntology(args.hpo_json)
    logger.info(f"Loaded HPO ontology with {len(ontology.data)} nodes from {args.hpo_json}.")

    # NER encoder + tokenizer (shared)
    tokenizer, model_tc, meta = load_ner_tc_and_tokenizer(
        args.backbone, args.init_encoder_from, args.model_dir, cfg
    )
    model_tc.to(device)
    for p in model_tc.parameters():
        p.requires_grad = False  # eval only

    hidden_size = model_tc.base.config.hidden_size

    # Span projection head
    ckpt_path = args.ckpt_path or os.path.join(args.out_dir, "hpoid_span_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    span_dim = ckpt.get("cfg", {}).get("hpoid_dim", 256)
    span_proj = SpanProj(in_dim=hidden_size, out_dim=span_dim, dropout=0.0).to(device)
    span_proj.load_state_dict(ckpt["span_proj_state"])
    span_proj.eval()
    logger.info(f"Loaded span projection head from {ckpt_path} (epoch={ckpt.get('epoch','?')}).")

    # Save eval config
    eval_cfg_path = os.path.join(args.out_dir, "hpoid_span_eval_config.json")
    with open(eval_cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cli": vars(args),
                "meta": meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved eval config to {eval_cfg_path}")

    # ---------- 结果收集：用于写 markdown & 绘图 ----------
    # results_summary[dataset] -> method -> metrics
    results_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Evaluate each dataset root separately
    global_step = 0
    for root in args.eval_roots:
        dataset_name = os.path.basename(root.rstrip("/"))
        logger.info(f"==== Evaluating dataset: {dataset_name} ====")

        ds = HPOIDSpanPairDataset(
            roots=[root],
            ontology=ontology,
            max_context_chars=256,
            max_syn=3,
        )

        if len(ds) == 0:
            logger.warning(f"[Eval] dataset={dataset_name} has no examples, skipping.")
            continue

        results_summary[dataset_name] = {}

        # -------------------------
        # 1) DualLoRAEnc only (FULL)
        # -------------------------
        reset_gpu_peak(device)
        t0 = time.time()
        metrics_ner = eval_full_retrieval(
            dataset_name,
            ds,
            model_tc,
            span_proj,
            tokenizer,
            ontology,
            cfg,
            device,
        )
        t1 = time.time()
        time_ner = t1 - t0
        gpu_ner = get_gpu_peak_mb(device)
        cpu_ner = get_cpu_mem_mb()
        logger.info(f"[DualLoRAEnc] dataset={dataset_name} done in {time_ner:.1f}s, "
                    f"GPU~{gpu_ner:.1f}MB, CPU~{cpu_ner:.1f}MB")

        writer.add_scalar(
            f"eval/{dataset_name}/ner/top1", metrics_ner["top1"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/ner/top5", metrics_ner["top5"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/ner/precision", metrics_ner["precision"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/ner/recall", metrics_ner["recall"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/ner/f1", metrics_ner["f1"], global_step
        )

        results_summary[dataset_name]["DualLoRAEnc"] = {
            "top1": float(metrics_ner["top1"]),
            "top5": float(metrics_ner["top5"]),
            "precision": float(metrics_ner["precision"]),
            "recall": float(metrics_ner["recall"]),
            "f1": float(metrics_ner["f1"]),
            "time_sec": float(time_ner),
            "gpu_mem_mb": float(gpu_ner),
            "cpu_mem_mb": float(cpu_ner),
            "recallA": None,
        }

        # -------------------------
        # 2) NER_FUZZY only PIPELINE_all
        # -------------------------
        reset_gpu_peak(device)
        t2 = time.time()
        metrics_fuzzy = eval_pipeline_AplusB(
            dataset_name,
            ds,
            model_tc,
            span_proj,
            tokenizer,
            ontology,
            cfg,
            device,
            stageA_topk=args.stageA_topk,
            stageA_mode="fuzzy_only",
            mode_tag="FUZZY_ONLY",
        )
        t3 = time.time()
        time_fuzzy = t3 - t2
        gpu_fuzzy = get_gpu_peak_mb(device)
        cpu_fuzzy = get_cpu_mem_mb()
        logger.info(f"[FUZZY_ONLY] dataset={dataset_name} done in {time_fuzzy:.1f}s, "
                    f"GPU~{gpu_fuzzy:.1f}MB, CPU~{cpu_fuzzy:.1f}MB")

        writer.add_scalar(
            f"eval/{dataset_name}/NER_FUZZY/recallA{args.stageA_topk}",
            metrics_fuzzy["recallA"],
            global_step,
        )
        writer.add_scalar(
            f"eval/{dataset_name}/NER_FUZZY/top1", metrics_fuzzy["top1"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/NER_FUZZY/top5", metrics_fuzzy["top5"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/NER_FUZZY/precision", metrics_fuzzy["precision"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/NER_FUZZY/recall", metrics_fuzzy["recall"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/NER_FUZZY/f1", metrics_fuzzy["f1"], global_step
        )

        results_summary[dataset_name]["FUZZY_ONLY"] = {
            "top1": float(metrics_fuzzy["top1"]),
            "top5": float(metrics_fuzzy["top5"]),
            "precision": float(metrics_fuzzy["precision"]),
            "recall": float(metrics_fuzzy["recall"]),
            "f1": float(metrics_fuzzy["f1"]),
            "time_sec": float(time_fuzzy),
            "gpu_mem_mb": float(gpu_fuzzy),
            "cpu_mem_mb": float(cpu_fuzzy),
            "recallA": float(metrics_fuzzy["recallA"]),
        }

        # -------------------------
        # 3) PIPELINE_all (all signals)
        # -------------------------
        reset_gpu_peak(device)
        t4 = time.time()
        metrics_pipe = eval_pipeline_AplusB(
            dataset_name,
            ds,
            model_tc,
            span_proj,
            tokenizer,
            ontology,
            cfg,
            device,
            stageA_topk=args.stageA_topk,
            stageA_mode="all",
            mode_tag="PIPELINE_all",
        )
        t5 = time.time()
        time_pipe = t5 - t4
        gpu_pipe = get_gpu_peak_mb(device)
        cpu_pipe = get_cpu_mem_mb()
        logger.info(f"[PIPELINE_all] dataset={dataset_name} done in {time_pipe:.1f}s, "
                    f"GPU~{gpu_pipe:.1f}MB, CPU~{cpu_pipe:.1f}MB")

        writer.add_scalar(
            f"eval/{dataset_name}/PIPELINE_all/recallA{args.stageA_topk}",
            metrics_pipe["recallA"],
            global_step,
        )
        writer.add_scalar(
            f"eval/{dataset_name}/PIPELINE_all/top1", metrics_pipe["top1"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/PIPELINE_all/top5", metrics_pipe["top5"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/PIPELINE_all/precision", metrics_pipe["precision"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/PIPELINE_all/recall", metrics_pipe["recall"], global_step
        )
        writer.add_scalar(
            f"eval/{dataset_name}/PIPELINE_all/f1", metrics_pipe["f1"], global_step
        )

        results_summary[dataset_name]["PIPELINE_all"] = {
            "top1": float(metrics_pipe["top1"]),
            "top5": float(metrics_pipe["top5"]),
            "precision": float(metrics_pipe["precision"]),
            "recall": float(metrics_pipe["recall"]),
            "f1": float(metrics_pipe["f1"]),
            "time_sec": float(time_pipe),
            "gpu_mem_mb": float(gpu_pipe),
            "cpu_mem_mb": float(cpu_pipe),
            "recallA": float(metrics_pipe["recallA"]),
        }

        global_step += 1

    writer.close()
    logger.info("Evaluation finished.")

    # ---------- 绘图 ----------
    plotter = EvalPlotter(args.out_dir)
    plot_paths = plotter.plot_all(results_summary)

    # ---------- 写 markdown summary 到 out_dir ----------
    md_path = os.path.join(args.out_dir, "hpoid_span_eval_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# HPO Span-Contrastive Eval Summary (DualLoRAEnc vs NER_FUZZY vs PIPELINE_all)\n\n")
        f.write("## Command\n\n")
        f.write("```bash\n")
        f.write("python eval_hpoid_span_contrastive.py \\\n")
        for k, v in vars(args).items():
            if isinstance(v, list):
                for item in v:
                    f.write(f"  --{k} {item} \\\n")
            else:
                f.write(f"  --{k} {v} \\\n")
        f.write("```\n\n")

        f.write("## Metrics per dataset\n\n")
        f.write(
            "| Dataset | Mode       | RecallA@{} | Top1 | Top5 | Precision | Recall | F1 | Time (s) | GPU MB | CPU MB |\n".format(
                args.stageA_topk
            )
        )
        f.write("|---------|-----------|-----------:|-----:|-----:|----------:|------:|----:|---------:|-------:|-------:|\n")

        for ds_name, methods in results_summary.items():
            # DualLoRAEnc only
            m_ner = methods.get("DualLoRAEnc", {})
            f.write(
                f"| {ds_name} | DualLoRAEnc  | {'-':>9} | "
                f"{m_ner.get('top1', 0.0):.4f} | {m_ner.get('top5', 0.0):.4f} | "
                f"{m_ner.get('precision', 0.0):.4f} | {m_ner.get('recall', 0.0):.4f} | {m_ner.get('f1', 0.0):.4f} | "
                f"{m_ner.get('time_sec', 0.0):.1f} | "
                f"{m_ner.get('gpu_mem_mb', -1.0):.1f} | {m_ner.get('cpu_mem_mb', -1.0):.1f} |\n"
            )

            # NER_FUZZY only
            m_fz = methods.get("FUZZY_ONLY", {})
            rec_fz = m_fz.get("recallA", None)
            rec_fz_str = "-" if rec_fz is None else f"{rec_fz:.4f}"
            f.write(
                f"| {ds_name} | NER_FUZZY     | {rec_fz_str:>9} | "
                f"{m_fz.get('top1', 0.0):.4f} | {m_fz.get('top5', 0.0):.4f} | "
                f"{m_fz.get('precision', 0.0):.4f} | {m_fz.get('recall', 0.0):.4f} | {m_fz.get('f1', 0.0):.4f} | "
                f"{m_fz.get('time_sec', 0.0):.1f} | "
                f"{m_fz.get('gpu_mem_mb', -1.0):.1f} | {m_fz.get('cpu_mem_mb', -1.0):.1f} |\n"
            )

            # PIPELINE_all
            m_pl = methods.get("PIPELINE_all", {})
            rec_pl = m_pl.get("recallA", None)
            rec_pl_str = "-" if rec_pl is None else f"{rec_pl:.4f}"
            f.write(
                f"| {ds_name} | PIPELINE_all  | {rec_pl_str:>9} | "
                f"{m_pl.get('top1', 0.0):.4f} | {m_pl.get('top5', 0.0):.4f} | "
                f"{m_pl.get('precision', 0.0):.4f} | {m_pl.get('recall', 0.0):.4f} | {m_pl.get('f1', 0.0):.4f} | "
                f"{m_pl.get('time_sec', 0.0):.1f} | "
                f"{m_pl.get('gpu_mem_mb', -1.0):.1f} | {m_pl.get('cpu_mem_mb', -1.0):.1f} |\n"
            )

        f.write("\n## Plots\n\n")
        if plot_paths.get("top1"):
            f.write(f"![Top-1 Accuracy]({os.path.basename(plot_paths['top1'])})\n\n")
        if plot_paths.get("top5"):
            f.write(f"![Top-5 Accuracy]({os.path.basename(plot_paths['top5'])})\n\n")
        if plot_paths.get("time"):
            f.write(f"![Time per method]({os.path.basename(plot_paths['time'])})\n\n")
        if plot_paths.get("gpu"):
            f.write(f"![GPU Memory]({os.path.basename(plot_paths['gpu'])})\n\n")
        if plot_paths.get("cpu"):
            f.write(f"![CPU Memory]({os.path.basename(plot_paths['cpu'])})\n\n")

    logger.info(f"Markdown summary saved to: {md_path}")


if __name__ == "__main__":
    main()


"""
python /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/full_eval_duallora.py \
  --eval_roots \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GeneReviews \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/GSC+ \
    /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/ID-68 \
  --hpo_json /cluster/home/gw/Backend_project/NER/pheno/PhenoBERT/phenobert/data/hpo.json \
  --model_dir /cluster/home/gw/Backend_project/NER/tuned/hpo_lora_onto_Dhead/best \
  --backbone /cluster/home/gw/Backend_project/models/BioLinkBERT-base \
  --init_encoder_from /cluster/home/gw/Backend_project/NER/tuned/intention \
  --out_dir /cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/VAL/full \
  --ckpt_path /cluster/home/gw/Backend_project/NER/tuned/hpoid_span_contrastive/hpoid_span_best.pt \
  --batch_size 32 \
  --max_len 512 \
  --stageA_topk 15
"""
