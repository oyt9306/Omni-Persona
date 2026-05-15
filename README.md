# Omni-Persona: Systematic Benchmarking and Improving Omnimodal Personalization

[![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon-b31b1b.svg)]() [![Project Page](https://img.shields.io/badge/Project-Page-blue)]()

We introduce **Omni-Persona**, the first comprehensive benchmark for **omnimodal personalization** spanning **text, image, and audio**. Built on the **Persona Modality Graph (PMG)**, it formalizes personalization as cross-modal routing and jointly evaluates **grounding** and **calibrated abstention** under realistic absent-persona retrieval noise.

## 🧑‍🔬 Authors

Yeongtak Oh, Dongwook Lee, Sangkwon Park, Heeseung Kim, Sungroh Yoon  
*Seoul National University · University of Seoul*

---

## 🔎 TL;DR

### 1️⃣ Task — Omnimodal Personalization

<p align="center">
  <img src="./data/figure1.jpg" alt="Figure 1. Formulation of omnimodal personalization." width="90%">
</p>

A user query arrives in **text / image / audio**. The model must (i) **identify the target persona** from pre-retrieved omnimodal contexts and (ii) **generate a context-grounded response**. We deliberately decouple retrieval from grounding to isolate the model's intrinsic expressiveness.

### 2️⃣ Formulation — Persona Modality Graph (PMG)

<p align="center">
  <img src="./data/PMG.jpg" alt="Figure 2. PMG illustration." width="50%">
</p>

Each persona is a node `(v, a, t)` (image, audio, text). Personalization becomes **cross-modal routing**: form an edge `e_{q→j}` to the matching node, or **abstain** if the persona is absent. This yields **4 task groups** (I2I, A2A, T2T, T2Any) and **18 fine-grained tasks**.

### 3️⃣ Evaluation — Answerable + Unanswerable

<p align="center">
  <img src="./data/omnimodal_context.png" alt="Figure 3. Context construction with answerable and unanswerable cases." width="90%">
</p>

~50% of the **750 items** are **unanswerable** (target persona absent), with **hard distractors** injected to stress grounding. We score with **Calibrated Accuracy** `Cal = ½ (Ans + Unans)`, jointly rewarding correct grounding *and* appropriate abstention.

---

## 🏋️ Training & 📊 Evaluation

> 🚧 **Coming Soon.** SFT (`ms-swift`) and RLVR (`TRL`) pipelines, the 750-item benchmark, and Cal / 1−FA / TA evaluation scripts will be released. Backbones: `Qwen2.5-Omni-3B/7B`, `Gemma4-E2B/E4B`.

---

## 📝 Cite

```bibtex
@article{oh2026omni,
  title={Omni-Persona: Systematic Benchmarking and Improving Omnimodal Personalization},
  author={Oh, Yeongtak and Lee, Dongwook and Park, Sangkwon and Kim, Heeseung and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2605.09996},
  year={2026}
}
```
