---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/LICENSE
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

**Out of scope.** These are research artifacts, not products. Do not deploy them for person
identification, biometric verification, speaker authentication, or any decision about a real
individual. They were trained on synthetic personas and are not validated on real-world identity
distributions. See [Limitations](#limitations).

## Training recipe

| Setting | Value |
|---|---|
| Algorithm | **GSPO** (RLVR, verifiable-reward policy optimization) |
| Reward mixture (`v20`) | **30 : 30 : 40** = localization : verification : text-QA |
| Outcome rewards | **asymmetric** — `r(TP) = r(TN) = +1.0`, `r(FN) = r(FP) = −0.5` |
| Checkpoint | **step 100** |
| Bias term | none (`nobias` variant) |
| Base | `Qwen/Qwen2.5-Omni-{3B,7B}` |
| Training data | synthetic personas (TTS voices + generated faces); the benchmark is held out |

The asymmetric reward is the design choice that drives the behaviour below. Correct answers and
correct abstentions are rewarded equally (+1.0), while both error types are penalized at half
magnitude (−0.5). Because a wrong *attempt* and a wrong *abstention* cost the same, and
attempting is the higher-expected-value action whenever the model has any signal, the policy
drifts toward answering.

> **Training code is not part of this release** ("coming soon"). This card documents the recipe
> so the released checkpoints are interpretable; it is not sufficient to re-run the training.

## Evaluation results

Omni-Persona v2.2, all 750 items, guarded protocol (an answerable item is correct only if the
model does **not** abstain **and** the judge returns `CORRECT`). All values ×100.

| Model | Ans ↑ | Cal ↑ | 1-FA ↑ | TA ↑ |
|---|---:|---:|---:|---:|
| Qwen2.5-Omni-3B (base) | 34.0 | 36.7 | 74.9 | 39.6 |
| **Qwen2.5-Omni-3B + RLVR (v20)** | **40.9** | 34.1 | 84.7 | 26.7 |
| Qwen2.5-Omni-7B (base) | 38.6 | 30.9 | 82.1 | 22.6 |
| **Qwen2.5-Omni-7B + RLVR (v20)** | **47.8** | 28.4 | 98.0 | 7.2 |

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

## Limitations

**RLVR raises answerable recall but does not improve calibrated accuracy.** This is the headline
caveat and it is visible in every row above:

- 3B: `Ans` +6.9 and `1-FA` +9.8, but `TA` −12.9 and `Cal` **−2.6**.
- 7B: `Ans` +9.2 and `1-FA` +15.9, but `TA` −15.4 and `Cal` **−2.5**.

The 7B checkpoint attempts **98.0%** of answerable items and abstains correctly on only **7.2%**
of unanswerable ones — it has very nearly stopped abstaining at all. The gain on `Ans` is
therefore partly a shift along the precision/abstention trade-off rather than better grounding.
**If your application needs a model that knows when to decline, the base checkpoints are the
better starting point.** These checkpoints are released to make that trade-off reproducible and
studyable, not because they dominate the baseline.

Other limitations:

- **English only.** Not evaluated in any other language.
- **Synthetic-to-real gap.** Training used synthetic faces and TTS voices; the benchmark uses
  real images and partly real speech. Behaviour on other real-world identity distributions is
  unmeasured.
- **Fixed context shape.** Tuned against 4 interleaved image+audio+text contexts in a single
  user turn with `temperature=0.0`, `max_tokens=256`. Longer or differently-shaped contexts are
  out of distribution.
- **Inherited base-model limitations.** All failure modes of Qwen2.5-Omni — hallucination, fine-
  grained speaker/face confusion, safety behaviour — carry over and were not separately
  mitigated.
- **No safety post-training.** The RLVR objective rewards benchmark correctness only. Assume the
  base model's safety alignment is *degraded*, not preserved.
- **Single seed, single evaluation set.** No variance estimates across seeds are reported here.

## License and attribution

⚠️ **Decision required before publishing.** These are derivative works of Qwen2.5-Omni. Both
`Qwen/Qwen2.5-Omni-3B` and `Qwen/Qwen2.5-Omni-7B` are published on the Hub under
`license: other` — a Qwen-specific license, **not** a blanket Apache-2.0 grant. Before release:

1. Read the `LICENSE`/`NOTICE` file in each base-model repo and confirm the exact license name,
   whether it is research-only or permits commercial use, and whether the 3B and 7B terms differ
   (they are distinct repos and may carry distinct terms).
2. Set `license_name` / `license_link` in the frontmatter of each model repo to the **verified**
   base-model license — do not leave the placeholder above.
3. Copy the base model's `LICENSE` file into each checkpoint repo, and keep any required
   attribution/naming notice.
4. State the derivative relationship in the repo body (already done via `base_model:`).

Base model citation:

```bibtex
@article{xu2025qwen2,
  title   = {Qwen2.5-Omni Technical Report},
  author  = {Qwen Team},
  journal = {arXiv preprint arXiv:2503.20215},
  year    = {2025}
}
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
