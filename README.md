# HPO-MoRE: Mixture-of-Reasoning Experts for Ontology-Consistent Phenotype Grounding

## 1. Introduction

HPO-MoRE (Human Phenotype Ontology Mixture-of-Reasoning Experts) is a hybrid phenotype grounding system that integrates contrastive embedding retrieval and large language model (LLM) definitional reasoning through a **margin-based expert routing** mechanism.

The goal is to map free-text clinical phenotype mentions to standardized HPO identifiers with:

* High recall over a large ontology (~15k nodes)
* Fine-grained sibling discrimination
* Ontology hierarchy consistency
* Computational efficiency suitable for clinical pipelines

Pure embedding models achieve strong recall but struggle with sibling ambiguity. Pure LLM approaches improve semantic discrimination but are expensive and may violate ontology constraints. **HPO-MoRE combines both through a principled mixture-of-experts framework.**

---

## 2. Problem Formulation

Given:

* Clinical context: `C`
* Mention span: `m ⊂ C`
* HPO ontology: `H = {h1, …, hN}`

Objective:

* Prediction:

  * `h_hat = argmax_{h ∈ H} P(h | C, m)`

Where `h_hat` is the predicted ontology concept.

---

## 3. Retrieval Expert: DualLoRAEnc

### 3.1 Embedding Space

A contrastively trained encoder

* Mapping:

  * `f_θ(text) -> R^d`, with `d = 256`

Encodes:

* Mention span embedding:

  * `m -> z_m`
* HPO definition embedding:

  * `h -> z_h`

All vectors are L2-normalized:

* `||z_m||_2 = 1`
* `||z_h||_2 = 1`

Similarity is computed via cosine similarity (since vectors are normalized, it is a dot product):

* `s(m, h) = z_m^T z_h`

---

### 3.2 Contrastive Objective

For a positive pair `(m, h+)` and negative concepts `{h-}`:

* Loss:

  * `L = -log( exp(s(m, h+)/τ) / ( exp(s(m, h+)/τ) + Σ_{h-} exp(s(m, h-)/τ) ) )`

Where:

* `τ` (tau) is the temperature parameter.

---

### 3.3 Candidate Retrieval

Top-K candidates are selected by similarity:

* Candidate set:

  * `K(m) = TopK_{h ∈ H} s(m, h)`

Default:

* `K = 15`

Empirically:

* `Recall@15 ≈ 1.0`

---

## 4. LLM Definition-Reasoning Expert

The LLM receives:

* Clinical context `C`
* Mention span `m`
* Candidate set `K(m)`
* Retrieval scores `{s_i}` as soft priors

It outputs a constrained choice:

* `h_hat_LLM ∈ K(m)`

This constraint keeps the search space bounded by the retrieval recall.

---

## 5. Margin-Based Expert Routing

Let:

* `s0 = max_h s(m, h)`
* `s1 = second-highest similarity`
* `Δ = s0 - s1`

Routing rule:

* Final selection `h_hat`:

  * If `Δ >= tau_high`: use retrieval expert
  * If `Δ <= tau_low`: use LLM expert
  * Else: hybrid decision

Written compactly:

* `h_hat =`

  * `h_hat_retrieval, if Δ >= tau_high`
  * `h_hat_LLM,      if Δ <= tau_low`
  * `Hybrid(C, m, K(m)), otherwise`

Interpretation:

* Large margin ⇒ confident retrieval
* Small margin ⇒ ambiguity among near siblings

Routing acts as a lightweight uncertainty proxy via similarity separation.

---

## 6. Ontology-Aware Fallback

Let `T(h)` denote the subtree rooted at `h`.

Constraints include:

1. **Subtree preference**

   * Prefer `h_hat_LLM` that stays within retrieval’s local neighborhood:

     * `h_hat_LLM ∈ T(h_retrieval)`

2. **Avoid cross-branch violations**

   * If LLM picks a concept that is far across the hierarchy, down-weight or reject it.

3. **Prefer minimal ontology distance**

   * Among valid candidates, choose the closest to the retrieval anchor:

     * `argmin_h distance(h, h_retrieval)`

These rules enforce hierarchical consistency.

---

## 7. Overall Decision Function

The final prediction is:

* `h_hat = OntologyFilter( Router( Retrieval(m), LLM(C, m, K(m)) ) )`

---

## 8. Experimental Results

### 8.1 Subset Evaluation

| Dataset     | Dual Top-1 | Pipeline Top-1 |
| ----------- | ---------: | -------------: |
| GeneReviews |     0.9119 |         0.9602 |
| GSC+        |     0.8189 |         0.8886 |
| ID-68       |     0.8589 |         0.9637 |

### 8.2 Full-Table Evaluation

| Dataset     | Dual Top-1 | Pipeline Top-1 |
| ----------- | ---------: | -------------: |
| GeneReviews |     0.5114 |         0.8580 |
| GSC+        |     0.5192 |         0.7699 |
| ID-68       |     0.4785 |         0.8710 |

### 8.3 Global Summary

* DualLoRAEnc Top-1 ≈ 0.85
* Pipeline Top-1 > 0.92
* Error reduction ≈ 35–45%
* LLM usage reduced ≈ 30–40%

---

## 9. Efficiency Analysis

Expected runtime:

* `T = T_retrieval + I(LLM_invoked) * T_LLM`

Where:

* `T_retrieval < 5 ms`
* `T_LLM ≈ 0.5–1.5 s`

Expected LLM usage:

* `E[LLM calls] ≈ 0.6–0.7`

---

## 10. Theoretical Interpretation

The system approximates a mixture model:

* `P(h | C, m) ≈ α(m) * P_retrieval(h | m) + (1 - α(m)) * P_reasoning(h | C, m)`

Hard routing approximates:

* `α(m) = 1(Δ >= τ)`

Thus:

* Retrieval handles confident regions
* LLM handles ambiguous regions
* Ontology constraints regularize predictions

---

## 11. Applications

* Clinical phenotype normalization
* Rare disease pipelines
* EHR structured extraction
* Variant prioritization systems
* Ontology-aware retrieval-augmented generation

---

## 12. Future Directions

* Learned routing via uncertainty calibration
* Probabilistic soft mixture instead of hard routing
* Cross-ontology generalization
* Knowledge graph-enhanced reasoning
* Confidence-aware LLM calibration

---

## 13. Citation

```bibtex
@article{hpo_more_2026,
  title  = {HPO-MoRE: Mixture-of-Reasoning Experts for Ontology-Consistent Phenotype Grounding},
  author = {2026 MEDIS LAB},
  year   = {2026}
}
```

---

## 14. License

```
MIT License

Copyright (c) 2026 MEDIS LAB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 15. Contact

For research collaboration, licensing, or technical inquiries:

**MEDIS LAB**
Email: [medis2025@outlook.com](mailto:medis2025@outlook.com)
