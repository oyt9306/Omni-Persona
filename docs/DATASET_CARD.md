---
task_categories:
  - question-answering
  - visual-question-answering
  - audio-classification
  - any-to-any
tags:
  - omnimodal
  - personalization
  - multimodal
  - abstention
  - calibration
  - benchmark
  - speaker-identification
  - face-identification
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: test
        path: omni_persona_v2_2.jsonl
---

# Omni-Persona Benchmark (v2.2)

Omni-Persona is a held-out **test set** for **omnimodal personalization**: given four
interleaved image + audio + text memory contexts about four different people, the model must
(i) route a query to the right person across modalities and (ii) answer a question about that
person **using only that person's context** — or **abstain** when the answer is not supported.

The benchmark is deliberately built so that a model cannot score well by being either
over-confident or over-cautious. Roughly half of the items are unanswerable.

- **750 items**, **391 answerable / 359 unanswerable**
- **4 task groups**, **18 fine-grained tasks**
- **250 source personas × 3 augmentations**
- Assets: **2,500 real files** (1,250 PNG + 1,250 WAV), **~1.2 GB**

---

## What it measures: the Persona Modality Graph (PMG)

Each persona is a node `(v, a, t)` — image, audio, text. Personalization is formalized as
**cross-modal routing**: given a query `q`, the model must either form an edge `e_{q→j}` to the
matching persona node `j` and read the requested attribute off that node, or determine that no
valid edge exists and **abstain**.

The query modality and the target attribute modality jointly define the routing path, which
yields four groups:

| Group | `scenario` value | Routing | Query is… | Items |
|---|---|---|---|---|
| **I2I** | `visual_identity` | image → identity → attribute | a face photo | 231 |
| **A2A** | `voice_identity` | audio → identity → attribute | a voice clip | 247 |
| **T2T** | `same_modal_semantic` | text → identity → attribute | a textual description | 124 |
| **T2Any** | `cross_modal_semantic_bridge` | text → *audio content* → identity → attribute | a description of what someone said | 148 |

T2Any is the hardest regime: the query describes conversational content that only exists inside
the context **audio**, with no explicit speaker cue, so identity must be resolved through a
text→audio semantic bridge before the attribute (which may itself live in a third modality) can
be read.

`scenario → group` is a frozen mapping; do not re-derive it:

```python
GROUP = {
    "visual_identity": "I2I",
    "voice_identity": "A2A",
    "same_modal_semantic": "T2T",
    "cross_modal_semantic_bridge": "T2Any",
}
```

## The 18 fine-grained tasks

Retrieval targets are `bio`, `dialogue`, `appearance`/`image`, `emotion`, `environment`.

| Group | Task (`selected_task`) | Total | Answerable | Unanswerable |
|---|---|---:|---:|---:|
| I2I | `visual_to_appearance` | 45 | 21 | 24 |
| I2I | `visual_to_bio` | 48 | 24 | 24 |
| I2I | `visual_to_dialogue` | 48 | 22 | 26 |
| I2I | `visual_to_emotion` | 45 | 27 | 18 |
| I2I | `visual_to_environment` | 45 | 21 | 24 |
| A2A | `voice_to_bio` | 53 | 22 | 31 |
| A2A | `voice_to_dialogue` | 56 | 27 | 29 |
| A2A | `voice_to_emotion` | 41 | 20 | 21 |
| A2A | `voice_to_environment` | 41 | 19 | 22 |
| A2A | `voice_to_image` | 56 | 22 | 34 |
| T2T | `semantic_to_appearance` | 36 | 22 | 14 |
| T2T | `semantic_to_bio` | 27 | 20 | 7 |
| T2T | `semantic_to_emotion` | 27 | 17 | 10 |
| T2T | `semantic_to_environment` | 34 | 24 | 10 |
| T2Any | `dialogue_to_bio` | 31 | 16 | 15 |
| T2Any | `dialogue_to_emotion` | 43 | 25 | 18 |
| T2Any | `dialogue_to_environment` | 36 | 16 | 20 |
| T2Any | `dialogue_to_image` | 38 | 26 | 12 |
| | **Total** | **750** | **391** | **359** |

T2T omits the `dialogue` target on purpose: text→text conversational matching collapses into
shallow keyword overlap.

## Answerable / unanswerable split

`answer_supervision.answerable_from_context` is the single source of truth. The 359
unanswerable items come in **two flavours**, and both are scored identically (correct ⇔ the model
abstains):

| Flavour | `no_GT` | `target_concept_id` | Meaning | Count |
|---|---|---|---|---|
| **Absent persona** | `true` | `null` | The queried person is not among the four contexts at all. | 150 |
| **Absent attribute** | `false` | set | The person *is* present, but the asked attribute is not stated anywhere in their context. | 209 |

Per group:

| Group | Total | Answerable | Unanswerable | …absent persona | …absent attribute |
|---|---:|---:|---:|---:|---:|
| I2I | 231 | 115 | 116 | 62 | 54 |
| A2A | 247 | 110 | 137 | 88 | 49 |
| T2T | 124 | 83 | 41 | 0 | 41 |
| T2Any | 148 | 83 | 65 | 0 | 65 |
| **All** | **750** | **391** | **359** | **150** | **209** |

The absent-attribute flavour is the harder half: the model must resolve identity correctly *and
then* recognise that the specific attribute is missing, rather than fabricating it from a
plausible neighbour context. I2I and A2A carry both flavours; T2T and T2Any carry only the
absent-attribute flavour by construction.

---

## Files

```
Omni-Persona-Benchmark/
├── omni_persona_v2_2.jsonl          # 750 annotation records (~1.7 MB)
├── assets/lsd/
│   ├── sample_0/
│   │   ├── concept_0.png  concept_0.wav
│   │   ├── concept_1.png  concept_1.wav
│   │   ├── concept_2.png  concept_2.wav
│   │   ├── concept_3.png  concept_3.wav
│   │   └── query.png      query.wav
│   ├── sample_1/ …
│   └── sample_249/                  # 250 dirs × 10 files = 2,500 files, ~1.2 GB
├── lsd_metadata.jsonl               # per-sample construction record (raw dialogues, labels)
└── gender_cache.json                # wav2vec2 gender predictions used to build distractors
```

Every `sample_<id>/` directory holds exactly 10 files (4 context images, 4 context audio clips,
1 query image, 1 query audio clip). Records reference assets by **paths relative to
`assets/lsd/`** — e.g. `"sample_0/concept_0.png"` — so pass that directory as `--asset-root`.
The 250 sample directories are shared by the 750 records (3 augmentations per source sample);
all 2,500 referenced assets are present, with no orphans and no missing files.

Because the assets are ~1.2 GB of small files, prefer a snapshot download over per-file `wget`:

```python
from huggingface_hub import snapshot_download
snapshot_download("Yeongtak/Omni-Persona-Benchmark", repo_type="dataset",
                  local_dir="omni-persona-data")
```

## Field schema

One JSON object per line.

| Field | Type | Read by eval? | Description |
|---|---|:--:|---|
| `augmented_id` | str | ✅ | Unique item id, `sample_<i>__aug_<j>`. Primary key / resume key. |
| `source_sample_id` | str | ✅ | Source persona bundle, `sample_<i>`. Also the asset directory name. |
| `scenario` | str | ✅ | One of the four scenario values above → task group. |
| `selected_task` | str | ✅ | One of the 18 fine-grained tasks. |
| `query_modality` | str | ✅ | `visual` \| `voice` \| `text_semantic` \| `dialogue_semantic`. **`voice` suppresses the query image** so identity cannot be resolved by face matching. |
| `target_modality` | str | ✅ | Requested attribute family: `bio`/`dialogue`/`image`/`emotion`/`environment`. |
| `no_GT` | bool | ✅ | `true` ⇔ the queried persona is absent from all four contexts. Reported as a slice; **not** the scoring flag. |
| `target_concept_id` | str \| null | – | Gold persona node (`concept_0..3`); `null` when `no_GT`. Useful for routing/localization analysis. |
| `query_assets` | obj | ✅ | `{image, audio}` relative paths for the query. |
| `contexts` | list[4] | ✅ | Four persona entries, always in a fixed order. |
| `contexts[].concept_id` | str | – | `concept_0..3`. |
| `contexts[].image` | str | ✅ | Relative path to the persona's face image. |
| `contexts[].audio` | str | ✅ | Relative path to the persona's voice clip. |
| `contexts[].text` | str | ✅ | 1–2 sentence persona profile. |
| `contexts[].same_as_query_image_or_audio` | bool | – | Oracle routing label: `true` for the persona matching the query. Exactly one `true` per item except for the 150 absent-persona items, which have none. **Diagnostic only — never feed it to the model.** |
| `query_text` | str | ✅ | The question. |
| `answer_supervision.answerable_from_context` | bool | ✅ | **The scoring flag.** |
| `answer_supervision.gold_answer` | str | ✅ | Reference answer; for all 359 unanswerable items it is the canonical abstention string. |
| `answer_supervision.abstain_answer` | str | – | The canonical abstention string, constant across the set. |
| `answer_supervision.support_mode` | str | ✅ | `present` (391) / `absent` (359); redundant with `answerable_from_context`, kept as a reporting slice. |
| `answer_supervision.asked_group` | str | ✅ | Attribute family asked. |
| `answer_supervision.asked_subfield` | str | ✅ | Finer attribute, e.g. `location`. |
| `answer_supervision.gt_concept_id_from_metadata` | str | – | Persona the query was generated from, including for absent-attribute items. |

Canonical abstention string (identical for all 750 items):

```
I cannot determine that from the provided context.
```

---

## Evaluation protocol

Two components produce a verdict per item.

**1. Abstention detection** — a frozen keyword rule (`omni_persona.judge.is_abstain`). The
prediction is lowercased and whitespace-collapsed, then substring-matched against:

```
cannot determine · cannot be determined · not enough information · insufficient information
cannot answer · unable to determine · don't know from · do not know from
the provided context does not · not provided in the context · no information in the context
context does not contain · cannot identify · i cannot tell
```

**2. LLM-as-a-judge** — `gpt-5.4-mini` over an OpenAI-compatible endpoint, comparing
`gold_answer` against the prediction. Parsing is strict: **only the exact token `CORRECT`
counts as correct**; anything else is wrong.

### Correctness rule

> **Answerable item is correct ⇔ `not is_abstain(pred)` AND `judge_verdict == "CORRECT"`.**
> **Unanswerable item is correct ⇔ `is_abstain(pred)`.**

The conjunction on the answerable side is what makes the metric *guarded*. A model that emits a
correct-looking answer wrapped in an abstention phrase gets no credit, and neither does a model
whose judge verdict is `CORRECT` for a hedged non-answer. This is the change from the v1 card.

### Metrics (all reported ×100)

| Metric | Definition |
|---|---|
| `Ans` | `ans_correct / n_answerable` — guarded answerable accuracy |
| `Unans` | `unans_abstain / n_unanswerable` — abstention accuracy (per scenario) |
| `Cal` | `(ans_correct + unans_abstain) / n_total` — **count-weighted** calibrated accuracy |
| `1-FA` | `(# answerable items where the model did NOT abstain) / n_answerable` — attempt rate |
| `TA` | `unans_abstain / n_unanswerable` — true abstention |
| `Avg` | `(1-FA + TA) / 2` |

`Cal` is computed over raw item counts, **not** as the macro average `½(Ans + Unans)`. With the
750-item split (391/359) the two differ; use the count-weighted form.

`1-FA` and `TA` depend only on the keyword rule, so they are deterministic. `Ans` and `Cal`
inherit judge sampling noise of up to ~1 point; treat a ≤1.0-point deviation as a reproduction.

### Prompt construction (frozen)

A single user turn: system prompt, then `=== Context 0..3 ===` blocks each carrying
image → audio → text in that order, then `=== Query ===` with the query image (**omitted when
`query_modality == "voice"`**), query audio, and `query_text`. Images go in as `image_url`
base64 data URIs, audio as `input_audio` WAV. Decoding defaults: `temperature=0.0`,
`max_tokens=256`.

## How to run the released evaluation

```bash
git clone https://github.com/oyt9306/Omni-Persona.git && cd Omni-Persona
pip install -r requirements.txt

hf download Yeongtak/Omni-Persona-Benchmark \
    --repo-type dataset --local-dir omni-persona-data

export BENCH=omni-persona-data/omni_persona_v2_2.jsonl
export ASSET_ROOT=omni-persona-data/assets/lsd
export JUDGE_BASE_URL=http://localhost:8091/v1   # OpenAI-compatible gpt-5.4-mini endpoint

scripts/smoke_test.sh Qwen/Qwen2.5-Omni-3B          # ~10 min, 40 stratified items
scripts/run_eval.sh   Qwen/Qwen2.5-Omni-7B qwen7b   # full 750, prints the main-table row
```

`run_eval.sh` serves the model with vLLM, runs inference, judges, scores, and tears the server
down. Results land in `results/<RUN_NAME>/`. See the repository README for the full env-var list.

---

## Source data and construction

- **Images** — sampled from the **CoViP evaluation split**, so that each individual recurs as a
  query subject across scenarios. Benchmark images are **real**, in contrast to the synthetic
  images used for training in the accompanying paper.
- **Text** — persona profiles and queries were synthesized from pre-generated dialogues with
  `GPT-5.4`: attribute extraction → plausible imputation of missing traits → 1–2 sentence
  profiles, then answerable/unanswerable query generation.
- **Audio** — a balanced mixture of synthetic and real speech over 450 distinct speakers.
  Synthetic clips come from [`chatterbox`](https://github.com/resemble-ai/chatterbox)
  (24-bit / 24 kHz); real 4–15 s conversational segments are curated from **VoxMM**, **MELD**,
  **JL-corpus**, and **RAVDESS**.
- **Distractors** — audio distractors are gender-matched using `wav2vec2` gender detection, so
  speaker identity cannot be shortcut through coarse gender cues. Visual distractors are chosen
  for perceptual similarity to the target.

## Intended use

Zero-shot evaluation of omnimodal (image + audio + text) assistants on grounded personalization
with calibrated abstention. **This is a held-out test set** — it is not a training corpus, and
fine-tuning on it invalidates any reported number.

## Citation

```bibtex
@article{oh2026omni,
  title   = {Omni-Persona: Systematic Benchmarking and Improving Omnimodal Personalization},
  author  = {Oh, Yeongtak and Lee, Dongwook and Park, Sangkwon and Kim, Heeseung and Yoon, Sungroh},
  journal = {arXiv preprint arXiv:2605.09996},
  year    = {2026}
}
```

Please also cite the upstream asset sources (CoViP, VoxMM, MELD, JL-corpus, RAVDESS) when using
the benchmark assets.
