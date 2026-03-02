#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_llm_refiner.py (LOCAL-LLM version, pipeline-adapted, +4bit)

- 支持从 prompt 文件加载完整提示词
- 使用 {{context}}, {{mention}}, {{candidates}} 作为变量
- ✅ 本地 Transformers HF 模型推理
- ✅ batch 方法：generate_indices_batch
- ✅ 新增 4bit quant（bitsandbytes）支持：load_in_4bit + BitsAndBytesConfig
"""

from typing import List, Dict, Any, Optional
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ NEW: 4bit quant support
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False


# ============================================================
# LOCAL LLM Client (with prompt_path support)
# ============================================================
class LLMAPIClient:
    """
    Local HF model client (keeps the same class name for pipeline compatibility).

    Supports:
        - local_model_dir (HF model path)
        - temperature=0 (deterministic)
        - prompt from external txt file
        - indices-only output
        - batch generation (generate_indices_batch)
        - ✅ 4-bit loading (bitsandbytes)
    """

    def __init__(
        self,
        api_key: str = "",   # unused in local mode
        base_url: str = "",  # unused in local mode
        model: str = "",     # unused in local mode
        timeout: float = 60.0,
        prompt_path: str = None,

        # ---- LOCAL-only args ----
        local_model_dir: str = None,
        local_max_new_tokens: int = 16,
        local_temperature: float = 0.0,
        local_max_input_tokens: int = 2048,
        trust_remote_code: bool = True,
        dtype: str = "bf16",     # "bf16" | "fp16" | "fp32"
        device_map: str = "auto",

        # ✅ NEW: 4-bit args (minimized additions)
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",         # "nf4" | "fp4"
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: str = "bf16",     # "bf16" | "fp16" | "fp32"
    ):
        self.api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout = timeout

        if not prompt_path or not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"prompt_path not found: {prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

        if not local_model_dir or not os.path.isdir(local_model_dir):
            raise FileNotFoundError(f"local_model_dir not found: {local_model_dir}")

        self.local_model_dir = local_model_dir
        self.local_max_new_tokens = int(local_max_new_tokens)
        self.local_temperature = float(local_temperature)
        self.local_max_input_tokens = int(local_max_input_tokens)

        # dtype choose (model compute dtype; for 4bit, this is usually bf16/fp16)
        dtype = (dtype or "bf16").lower()
        if dtype == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # ✅ NEW: bnb compute dtype
        bnb_4bit_compute_dtype = (bnb_4bit_compute_dtype or "bf16").lower()
        if bnb_4bit_compute_dtype == "bf16":
            bnb_compute_dtype = torch.bfloat16
        elif bnb_4bit_compute_dtype == "fp16":
            bnb_compute_dtype = torch.float16
        else:
            bnb_compute_dtype = torch.float32

        self.load_in_4bit = bool(load_in_4bit)
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bool(bnb_4bit_use_double_quant)
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

        # Some models lack pad_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ✅ IMPORTANT: true batch for decoder-only needs LEFT padding
        # (won't harm non-chat models; helps for generate_indices_batch)
        self.tokenizer.padding_side = "left"

        # ✅ NEW: quantization config
        quant_config = None
        if self.load_in_4bit:
            if not _HAS_BNB or BitsAndBytesConfig is None:
                raise RuntimeError(
                    "load_in_4bit=True but BitsAndBytesConfig is unavailable. "
                    "Please install bitsandbytes + a compatible transformers build."
                )
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
            )

        # Model load
        # NOTE: when using quantization_config, torch_dtype is still OK to pass (controls some modules)
        self.model_hf = AutoModelForCausalLM.from_pretrained(
            local_model_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        )
        self.model_hf.eval()

    # ---------------------------------------------------------
    # Build final prompt
    # ---------------------------------------------------------
    def build_prompt(self, context: str, mention: str, candidates_block: str) -> str:
        p = self.prompt_template
        p = p.replace("{{context}}", context or "")
        p = p.replace("{{mention}}", mention or "")
        p = p.replace("{{candidates}}", candidates_block or "")
        return p

    # ---- internal: build input text for chat-template models ----
    def _maybe_wrap_chat(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    # ✅ NEW: pick an input device that matches the model's first stage (embedding device).
    # model_hf.device is unreliable with device_map="auto" (sharded model).
    def _get_input_device(self) -> Optional[torch.device]:
        # If model is on single device, this works
        try:
            if hasattr(self.model_hf, "device") and isinstance(self.model_hf.device, torch.device):
                return self.model_hf.device
        except Exception:
            pass

        # If sharded, find embed_tokens device (most reliable)
        try:
            emb = self.model_hf.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight") and emb.weight is not None:
                return emb.weight.device
        except Exception:
            pass

        # Fallback: try hf_device_map first entry
        try:
            dm = getattr(self.model_hf, "hf_device_map", None)
            if isinstance(dm, dict) and len(dm) > 0:
                # choose the first mapped device that is cuda/xpu/cpu
                for _, dev_str in dm.items():
                    if isinstance(dev_str, str) and dev_str:
                        return torch.device(dev_str)
        except Exception:
            pass

        return None

    def _move_enc_to_model_input_device(self, enc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dev = self._get_input_device()
        if dev is None:
            return enc
        try:
            return {k: v.to(dev) for k, v in enc.items()}
        except Exception:
            return enc

    # ---------------------------------------------------------
    # Local generation (single)
    # ---------------------------------------------------------
    @torch.inference_mode()
    def generate_indices_only(self, prompt: str) -> str:
        """
        Local HF generation.
        Return raw text (should be like: "0" / "1,2" / "-1").
        """
        rendered = self._maybe_wrap_chat(prompt)
        enc = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.local_max_input_tokens,
        )

        enc = self._move_enc_to_model_input_device(enc)

        do_sample = self.local_temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.local_max_new_tokens,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.local_temperature

        out = self.model_hf.generate(**enc, **gen_kwargs)

        prompt_len = enc["input_ids"].shape[-1]
        gen_ids = out[0][prompt_len:]
        decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return (decoded or "").strip()

    # ---------------------------------------------------------
    # Local generation (true batch)
    # ---------------------------------------------------------
    @torch.inference_mode()
    def generate_indices_batch(self, prompts: List[str]) -> List[str]:
        """
        True batched generation: multiple prompts -> multiple raw outputs.
        Each output should be like: "0" / "1,2" / "-1".
        """
        if not prompts:
            return []

        rendered_list = [self._maybe_wrap_chat(p) for p in prompts]

        enc = self.tokenizer(
            rendered_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.local_max_input_tokens,
        )

        enc = self._move_enc_to_model_input_device(enc)

        do_sample = self.local_temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=self.local_max_new_tokens,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.local_temperature

        out = self.model_hf.generate(**enc, **gen_kwargs)

        # For batch with padding: use attention_mask lengths per row
        input_lens = enc["attention_mask"].sum(dim=1).tolist()

        raws: List[str] = []
        for i in range(len(prompts)):
            gen_ids = out[i][int(input_lens[i]):]
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            raws.append((decoded or "").strip())

        return raws


# ============================================================
# HPO Candidate Refiner
# ============================================================
class HPOCandidateRefiner:
    """
    LLM-based refinement for HPO candidates.

    Only outputs index list: [0], [0,2], or [].

    ✅ Added:
        - refine_batch()        (recommended, concurrency)
        - refine_batch_true()   (true GPU batching)
    """

    def __init__(self, llm_client: LLMAPIClient, max_candidates: int = 10):
        self.llm = llm_client
        self.max_candidates = max_candidates

    def _build_candidates_list(self, candidates: List[Dict[str, Any]]) -> str:
        lines = []
        for idx, c in enumerate(candidates[: self.max_candidates]):
            hid = c.get("hpo_id", "")
            name = c.get("hpo_name", "")

            definition = c.get("hpo_def")
            if definition is None and "Def" in c:
                definition = c["Def"]
            if isinstance(definition, list):
                definition = definition[0] if definition else ""
            if not isinstance(definition, str):
                definition = ""

            syns = c.get("hpo_synonyms") or c.get("Synonym") or []
            if isinstance(syns, list):
                syn_str = "; ".join([str(x) for x in syns])
            else:
                syn_str = str(syns)

            block = [
                f"[{idx}]",
                f"  ID: {hid}",
                f"  Name: {name}",
                f"  Definition: {definition}",
                f"  Synonyms: {syn_str}",
            ]
            lines.append("\n".join(block))

        return "\n\n".join(lines)

    @staticmethod
    def _parse_indices(raw: str, num_candidates: int) -> List[int]:
        raw = (raw or "").strip()
        allowed = set("0123456789,-")

        if any(ch not in allowed for ch in raw):
            return []

        if raw in ("", "-1"):
            return []

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                idx = int(p)
            except Exception:
                continue
            if 0 <= idx < num_candidates:
                out.append(idx)

        seen = set()
        dedup: List[int] = []
        for i in out:
            if i not in seen:
                seen.add(i)
                dedup.append(i)
        return dedup

    def refine(self, context: str, mention: str, candidates: List[Dict[str, Any]]):
        if not candidates:
            return []

        cblock = self._build_candidates_list(candidates)
        prompt = self.llm.build_prompt(
            context=context,
            mention=mention,
            candidates_block=cblock,
        )
        raw = self.llm.generate_indices_only(prompt)
        return self._parse_indices(
            raw,
            num_candidates=min(len(candidates), self.max_candidates),
        )

    def refine_batch(
        self,
        batch: List[Dict[str, Any]],
        max_workers: int = 2,
        show_progress: bool = False,
    ) -> List[List[int]]:
        if not batch:
            return []

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[Optional[List[int]]] = [None] * len(batch)

        def _run(i: int, item: Dict[str, Any]) -> None:
            results[i] = self.refine(
                context=item.get("context", "") or "",
                mention=item.get("mention", "") or "",
                candidates=item.get("candidates", []) or [],
            )

        idx_iter = range(len(batch))
        if show_progress:
            try:
                from tqdm import tqdm
                idx_iter = tqdm(list(idx_iter), total=len(batch))
            except Exception:
                idx_iter = range(len(batch))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_run, i, batch[i]) for i in idx_iter]
            for f in as_completed(futs):
                f.result()

        return [r or [] for r in results]

    def refine_batch_true(self, batch: List[Dict[str, Any]]) -> List[List[int]]:
        if not batch:
            return []

        prompts: List[str] = []
        num_cands: List[int] = []

        for item in batch:
            cands = item.get("candidates", []) or []
            cblock = self._build_candidates_list(cands)
            prompt = self.llm.build_prompt(
                context=item.get("context", "") or "",
                mention=item.get("mention", "") or "",
                candidates_block=cblock,
            )
            prompts.append(prompt)
            num_cands.append(min(len(cands), self.max_candidates))

        raws = self.llm.generate_indices_batch(prompts)

        out: List[List[int]] = []
        for raw, n in zip(raws, num_cands):
            out.append(self._parse_indices(raw, num_candidates=n))
        return out


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    PROMPT_FILE = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/prompts/select_hpo.txt"
    LOCAL_MODEL_DIR = "/cluster/home/gw/Backend_project/models/Qwen2.5-Aloe-Beta-7B"

    client = LLMAPIClient(
        prompt_path=PROMPT_FILE,
        local_model_dir=LOCAL_MODEL_DIR,
        local_max_new_tokens=16,
        local_temperature=0.0,
        local_max_input_tokens=2048,
        dtype="bf16",
        device_map="auto",

        # ✅ 4bit on
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bf16",
    )

    refiner = HPOCandidateRefiner(client)

    context = "The patient has spinning sensation when standing, consistent with dizziness."
    mention = "spinning sensation"
    candidates = [
        {
            "hpo_id": "HP:0002321",
            "hpo_name": "Vertigo",
            "hpo_def": "A sensation of spinning.",
            "hpo_synonyms": ["Dizziness"],
        },
        {
            "hpo_id": "HP:0001250",
            "hpo_name": "Seizures",
            "hpo_def": "Transient abnormal brain activity.",
            "hpo_synonyms": ["Epileptic seizures"],
        }
    ]
    print("Single selected indices:", refiner.refine(context, mention, candidates))

    batch_inputs = [
        {"context": context, "mention": mention, "candidates": candidates},
        {"context": "Patient has generalized tonic-clonic episodes.", "mention": "tonic-clonic episodes", "candidates": candidates},
    ]
    print("Batch selected indices (true GPU batch):", refiner.refine_batch_true(batch_inputs))
