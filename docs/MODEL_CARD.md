---
base_model:
  - Qwen/Qwen2.5-Omni-7B
language:
  - en
library_name: transformers
pipeline_tag: any-to-any
tags:
  - omnimodal
  - personalization
  - rlvr
  - gspo
  - abstention
  - qwen2_5_omni
---

<!--
This card covers BOTH released checkpoints. When publishing, split it into two HF model repos
and keep only the matching row of the results table + the matching `base_model` field:
  Yeongtak/Qwen2.5-Omni-3B-Omni-Persona-RLVR  <- base_model: Qwen/Qwen2.5-Omni-3B
  Yeongtak/Qwen2.5-Omni-7B-Omni-Persona-RLVR  <- base_model: Qwen/Qwen2.5-Omni-7B
-->

# Omni-Persona RLVR checkpoints (Qwen2.5-Omni-3B / 7B)

Two omnimodal checkpoints post-trained with **RLVR** for grounded personalization on the
[Omni-Persona benchmark](https://huggingface.co/datasets/Yeongtak/Omni-Persona-Benchmark).
Both are **fully merged weights** — no adapter, load them like the base model.

| Release | Base | Params | Size on disk | Files |
|---|---|---:|---:|---|
| `Qwen2.5-Omni-3B-Omni-Persona-RLVR` | `Qwen/Qwen2.5-Omni-3B` | 5.5 B | 12 GB | 3 safetensors shards |
| `Qwen2.5-Omni-7B-Omni-Persona-RLVR` | `Qwen/Qwen2.5-Omni-7B` | 10.7 B | 21 GB | 5 safetensors shards |

Both ship the full omni preprocessor stack (`preprocessor_config.json`,
`video_preprocessor_config.json`, `chat_template.jinja`), so they drop into the same vLLM and
`transformers` paths as the corresponding base model.

## Intended use

Research on **omnimodal personalization**: routing a query to the right persona across image,
audio and text memory contexts, then answering from that persona's context or abstaining when
the answer is not supported. Evaluated zero-shot on the Omni-Persona 750-item test set.

## Training recipe

| Setting | Value |
|---|---|
| Algorithm | **GSPO** (RLVR, verifiable-reward policy optimization) |
| Reward mixture (`v20`) | **30 : 30 : 40** = localization : verification : text-QA |
| Outcome rewards | **asymmetric** — `r(TP) = r(TN) = +1.0`, `r(FN) = r(FP) = −0.5` |
| Checkpoint | **step 100** |
| Bias term | none (`nobias` variant) |
| Base | `Qwen/Qwen2.5-Omni-{3B,7B}` |

## Evaluation results

Omni-Persona v2.2, all 750 items, guarded protocol (an answerable item is correct only if the
model does **not** abstain **and** the judge returns `CORRECT`). All values ×100.

| Model | Ans ↑ | Cal ↑ | 1-FA ↑ | TA ↑ |
|---|---:|---:|---:|---:|
| Qwen2.5-Omni-3B (base) | 34.0 | 36.7 | 74.9 | 39.6 |
| **Qwen2.5-Omni-3B + RLVR ** | **40.9** | 34.1 | 84.7 | 26.7 |
| Qwen2.5-Omni-7B (base) | 38.6 | 30.9 | 82.1 | 22.6 |
| **Qwen2.5-Omni-7B + RLVR ** | **47.8** | 28.4 | 98.0 | 7.2 |

- `Ans` — guarded accuracy on the 391 answerable items
- `Cal` — count-weighted calibrated accuracy over all 750 items
- `1-FA` — attempt rate on answerable items (1 − false-abstention rate)
- `TA` — true-abstention rate on the 359 unanswerable items

Judge non-determinism moves `Ans`/`Cal` by up to ~1 point; `1-FA` and `TA` are keyword-based and
reproduce near-exactly.

Reproduce with the released evaluation code:

```bash
bash scripts/serve_model.sh <path-or-hub-id-of-this-checkpoint>
bash scripts/run_eval.sh \
    --data omni-persona-data/omni_persona_v2_2.jsonl \
    --asset-root omni-persona-data/assets/lsd \
    --model <served-model-name> \
    --judge-base-url http://localhost:8091/v1
```

## Citation

```bibtex
@article{oh2026omni,
  title   = {Omni-Persona: Systematic Benchmarking and Improving Omnimodal Personalization},
  author  = {Oh, Yeongtak and Lee, Dongwook and Park, Sangkwon and Kim, Heeseung and Yoon, Sungroh},
  journal = {arXiv preprint arXiv:2605.09996},
  year    = {2026}
}
```
