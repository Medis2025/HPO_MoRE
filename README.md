# HPO-MoRE: Mixture-of-Reasoning Experts for Ontology-Consistent Phenotype Grounding

## 1. Introduction

HPO-MoRE (Human Phenotype Ontology Mixture-of-Reasoning Experts) is a hybrid phenotype grounding system that integrates contrastive embedding retrieval and large language model (LLM) definitional reasoning through a margin-based expert routing mechanism.

The goal is to map free-text clinical phenotype mentions to standardized HPO identifiers with:

* High recall over a large ontology (~15k nodes)
* Fine-grained sibling discrimination
* Ontology hierarchy consistency
* Computational efficiency suitable for clinical pipelines

Pure embedding models achieve strong recall but struggle with sibling ambiguity. Pure LLM approaches improve semantic discrimination but are expensive and may violate ontology constraints. HPO-MoRE combines both through a principled mixture-of-experts framework.

---

## 2. Problem Formulation

Given:

* Clinical context ( C )
* Mention span ( m \subset C )
* HPO ontology ( \mathcal{H} = {h_1, \dots, h_N} )

The objective is:

[
\hat{h} = \arg\max_{h \in \mathcal{H}} P(h \mid C, m)
]

where ( \hat{h} ) is the predicted ontology concept.

---

## 3. Retrieval Expert: DualLoRAEnc

### 3.1 Embedding Space

A contrastively trained encoder

[
f_\theta : \text{text} \rightarrow \mathbb{R}^{d}, \quad d = 256
]

maps:

* Mention spans ( m \rightarrow \mathbf{z}_m )
* HPO definitions ( h \rightarrow \mathbf{z}_h )

All vectors are L2-normalized:

[
|\mathbf{z}_m|_2 = |\mathbf{z}_h|_2 = 1
]

Similarity is computed via cosine similarity:

[
s(m, h) = \mathbf{z}_m^\top \mathbf{z}_h
]

---

### 3.2 Contrastive Objective

For a positive pair ((m, h^+)) and negatives (h^-):

[
\mathcal{L} = -\log \frac{\exp(s(m, h^+)/\tau)}
{\exp(s(m, h^+)/\tau) + \sum_{h^-} \exp(s(m, h^-)/\tau)}
]

where ( \tau ) is a temperature parameter.

---

### 3.3 Candidate Retrieval

Top-K candidates are selected:

[
\mathcal{K}(m) = \text{TopK}_{h \in \mathcal{H}} ; s(m,h)
]

Default: ( K = 15 ).

Empirically:

[
\text{Recall@15} \approx 1.0
]

---

## 4. LLM Definition-Reasoning Expert

The LLM receives:

* Clinical context ( C )
* Mention span ( m )
* Candidate set ( \mathcal{K}(m) )
* Retrieval scores ( {s_i} ) as soft priors

It outputs:

[
\hat{h}_{LLM} \in \mathcal{K}(m)
]

Constrained selection ensures search space remains bounded by retrieval recall.

---

## 5. Margin-Based Expert Routing

Let:

[
s_0 = \max_{h} s(m,h)
]

[
s_1 = \text{second-highest similarity}
]

[
\Delta = s_0 - s_1
]

Routing rule:

[
\hat{h} =
\begin{cases}
\hat{h}*{retrieval} & \text{if } \Delta \ge \tau*{high} \
\hat{h}*{LLM} & \text{if } \Delta \le \tau*{low} \
\text{Hybrid decision} & \text{otherwise}
\end{cases}
]

Interpretation:

* Large margin implies confident retrieval
* Small margin implies ambiguity

Routing approximates uncertainty estimation via similarity separation.

---

## 6. Ontology-Aware Fallback

Let ( \mathcal{T}(h) ) denote the subtree rooted at ( h ).

Constraints include:

1. Subtree preference:

[
\hat{h}*{LLM} \in \mathcal{T}(h*{retrieval})
]

2. Avoid cross-branch violations

3. Prefer minimal ontology distance:

[
\arg\min_h ; \text{distance}(h, h_{retrieval})
]

These rules enforce hierarchical consistency.

---

## 7. Overall Decision Function

The final prediction is:

[
\hat{h} = \text{OntologyFilter}
\left(
\text{Router}
\left(
\text{Retrieval}(m),
\text{LLM}(m, \mathcal{K}(m))
\right)
\right)
]

---

## 8. Experimental Results

### 8.1 Subset Evaluation

| Dataset     | Dual Top-1 | Pipeline Top-1 |
| ----------- | ---------: | -------------: |
| GeneReviews |     0.9119 |         0.9602 |
| GSC+        |     0.8189 |         0.8886 |
| ID-68       |     0.8589 |         0.9637 |

---

### 8.2 Full-Table Evaluation

| Dataset     | Dual Top-1 | Pipeline Top-1 |
| ----------- | ---------: | -------------: |
| GeneReviews |     0.5114 |         0.8580 |
| GSC+        |     0.5192 |         0.7699 |
| ID-68       |     0.4785 |         0.8710 |

---

### 8.3 Global Summary

* DualLoRAEnc Top-1 ≈ 0.85
* Pipeline Top-1 > 0.92
* Error reduction ≈ 35–45%
* LLM usage reduced ≈ 30–40%

---

## 9. Efficiency Analysis

Expected runtime:

[
T = T_{retrieval} + \mathbb{I}(\text{LLM invoked}) \cdot T_{LLM}
]

Where:

[
T_{retrieval} < 5 \text{ ms}
]

[
T_{LLM} \approx 0.5 - 1.5 \text{ s}
]

Expected LLM usage:

[
\mathbb{E}[\text{LLM calls}] \approx 0.6 - 0.7
]

---

## 10. Theoretical Interpretation

The system approximates:

[
P(h \mid C,m)
=============

\alpha(m) P_{retrieval}(h \mid m)
+
(1 - \alpha(m)) P_{reasoning}(h \mid C,m)
]

Routing approximates:

[
\alpha(m) = \mathbf{1}(\Delta \ge \tau)
]

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

```
@article{hpo_more_2026,
  title={HPO-MoRE: Mixture-of-Reasoning Experts for Ontology-Consistent Phenotype Grounding},
  author={2026 MEDIS LAB},
  year={2026}
}
```

---

## 14. License

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

---

## 15. Contact

For research collaboration, licensing, or technical inquiries:

MEDIS LAB
Email: [medis2025@outlook.com](mailto:medis2025@outlook.com)
