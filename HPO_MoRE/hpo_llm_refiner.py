#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hpo_llm_refiner.py (pipeline-adapted version)

- 支持从 prompt 文件加载完整提示词
- 使用 {{context}}, {{mention}}, {{candidates}} 作为变量
- 与 pipeline 中 HPOPipeline 配合无缝
"""

from typing import List, Dict, Any
import requests
import os


# ============================================================
# LLM API Client (with prompt_path support)
# ============================================================
class LLMAPIClient:
    """
    Minimal LLMapi-style client.

    Supports:
        - base_url
        - model
        - temperature=0
        - prompt from external txt file
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 60.0,
        prompt_path: str = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        if not prompt_path or not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

    # Build final prompt
    def build_prompt(self, context: str, mention: str, candidates_block: str) -> str:
        p = self.prompt_template
        p = p.replace("{{context}}", context or "")
        p = p.replace("{{mention}}", mention or "")
        p = p.replace("{{candidates}}", candidates_block)
        return p

    # Raw call
    def generate_indices_only(self, prompt: str) -> str:
        """
        Call DeepSeek with the given prompt (already rendered).
        Return raw text (should be like: "0" / "1,2" / "-1").
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 16,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ============================================================
# HPO Candidate Refiner
# ============================================================
class HPOCandidateRefiner:
    """
    LLM-based refinement for HPO candidates.

    Input candidates recommended format:
        {
            "hpo_id": "...",
            "hpo_name": "...",
            "hpo_def": "...",
            "hpo_synonyms": [...],
            "score": float
        }

    Only outputs index list: [0], [0,2], or [].
    """

    def __init__(self, llm_client: LLMAPIClient, max_candidates: int = 10):
        self.llm = llm_client
        self.max_candidates = max_candidates

    # ---------------------------------------------------------
    # Build candidates block for prompt
    # ---------------------------------------------------------
    def _build_candidates_list(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Render candidates to readable block for prompt:

        [0]
            ID: HP:0002321
            Name: Vertigo
            Definition: ...
            Synonyms: a; b; c
        """
        lines = []
        for idx, c in enumerate(candidates[: self.max_candidates]):
            hid = c.get("hpo_id", "")
            name = c.get("hpo_name", "")

            # Definition
            definition = c.get("hpo_def")
            if definition is None and "Def" in c:
                definition = c["Def"]
            if isinstance(definition, list):
                definition = definition[0] if definition else ""
            if not isinstance(definition, str):
                definition = ""

            # Synonyms
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

    # ---------------------------------------------------------
    # Parse LLM output
    # ---------------------------------------------------------
    @staticmethod
    def _parse_indices(raw: str, num_candidates: int) -> List[int]:
        allowed = set("0123456789,-")
        if any(ch not in allowed for ch in raw.strip()):
            return []

        if raw.strip() in ("", "-1"):
            return []

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                idx = int(p)
            except:
                continue
            if 0 <= idx < num_candidates:
                out.append(idx)

        # dedupe
        seen = set()
        dedup = []
        for i in out:
            if i not in seen:
                seen.add(i)
                dedup.append(i)
        return dedup

    # ---------------------------------------------------------
    # Core refine
    # ---------------------------------------------------------
    def refine(self, context: str, mention: str, candidates: List[Dict[str, Any]]):
        if not candidates:
            return []

        cblock = self._build_candidates_list(candidates)

        # render prompt from file
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


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    # Example:
    API_KEY = "sk-7dcfd726a1804d3e9c3703538d9cd0c3"

    PROMPT_FILE = "/cluster/home/gw/Backend_project/NER/process_bert/add_head/HPO/NER_H/pipeline_remote/prompts/select_hpo.txt"

    client = LLMAPIClient(
        api_key=API_KEY,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        timeout=60,
        prompt_path=PROMPT_FILE,
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

    result = refiner.refine(context, mention, candidates)
    print("Selected indices:", result)
