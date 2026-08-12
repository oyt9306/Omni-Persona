# Omni-Persona RLVR — training data and skeleton

Reinforcement learning with verifiable rewards for **Qwen2.5-Omni**. This is the
reference implementation of the recipe behind the released
[3B](https://huggingface.co/Yeongtak/Qwen2.5-Omni-3B-Omni-Persona-RLVR) and
[7B](https://huggingface.co/Yeongtak/Qwen2.5-Omni-7B-Omni-Persona-RLVR)
checkpoints.

---

## What is here

```
training/
├── omni_persona_rlvr/
│   ├── reward.py      three verifiable reward components
│   ├── dataset.py     30:30:40 task mixture, 1:0.6 answerable ratio
│   └── train.py       GSPO entry point (TRL)
├── scripts/
│   └── train_rlvr_qwen.sh
└── data/
    ├── rl_pairs.json          382 persona query/answer pairs (text_qa)
    └── localize_verify.jsonl  1,828 localize / verify prompts
```

`data/` carries the **prompts and labels only**. The images and audio they point
at are the benchmark assets; see [Assets](#assets).

---

## The reward

Three components, one per capability the benchmark isolates. Each reduces a
completion to one of four verifiable outcomes:

| | meaning |
|---|---|
| **TP** | a correct grounded answer, or a correct match |
| **TN** | a correct abstention, or a correct rejection |
| **FN** | a false abstention, or a missed match |
| **FP** | an unsupported answer, or a false match |

scored as

```
r(TP) = r(TN) = +1.0        r(FN) = r(FP) = -0.5
```

with every remaining outcome — an incorrect grounded answer, an unparsable
response, a wrong concrete index — scoring **0**.

Rewarding TP and TN equally stops the policy from inflating its apparent
grounding by answering everything, while the smaller penalty on FN/FP keeps a
single mistake from dominating a rollout group. The judge is binary: an answer
is correct or it is not, with no partial credit.

Concretely:

| Component | TP | TN | FN | FP |
|---|---|---|---|---|
| **localize** | correct context index | correct "none" | said "none", a context matched | named a context, none matched |
| **verify** | label yes, predicted yes | label no, predicted no | label yes, predicted no | label no, predicted yes |
| **text_qa** | answerable, judge says correct | unanswerable, abstained | answerable, abstained | unanswerable, answered anyway |

Every positive reward is gated on a degeneration check (4-gram repetition,
character diversity, repeated sentences), so a repetitive completion that happens
to contain the right token earns nothing.

The four values are overridable through `RLVR_R_TP` / `RLVR_R_TN` / `RLVR_R_FN` / `RLVR_R_FP`.

---

## The mixture

```
localize 30%  :  verify 30%  :  text_qa 40%
```

`localize` and `verify` each split evenly across their audio and image variants,
giving five concrete task types. The `text_qa` pool is held at a **1 : 0.6**
answerable-to-unanswerable ratio, i.e. roughly 37.5% of `text_qa` draws are
absent-persona items that the model should decline.

---

## Running it

```bash
pip install -r training/requirements.txt
```

The `text_qa` reward needs an LLM judge behind an OpenAI-compatible endpoint. The
run aborts up front if it does not answer — a silently dead judge makes every
`text_qa` reward `0.0`, and the run then looks like it is merely learning slowly.

```bash
export DATA_ROOT=/path/to/assets          # holds Benchmark_tot/ and query media
export JUDGE_BASE_URL=http://localhost:8091/v1
export JUDGE_MODEL=gpt-5.4-mini

MODEL=Qwen/Qwen2.5-Omni-3B CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/train_rlvr_qwen.sh
```

Configuration behind the released checkpoints:

| | |
|---|---|
| Algorithm | GSPO (GRPO with `importance_sampling_level="sequence"`) |
| Steps | 100 (checkpoint every 50) |
| Rollouts per prompt | 8 |
| Sampling temperature | 1.1 |
| KL coefficient β | 0.04 |
| Clipping ε / ε_high | 0.2 / 0.2 |
| Max completion length | 512 |
| Effective batch | 8 (`grad_accum` scales with GPU count) |
| Precision | bf16, gradient checkpointing, FlashAttention-2 |

Only the Thinker is trained. Qwen2.5-Omni's Talker is a speech decoder that takes
no gradient from a text-only objective, and loading it only costs memory.

**Note on DeepSpeed ZeRO-2:** gradient accumulation across multiple GPUs is not
usable with this setup. Run either one GPU with `GRAD_ACCUM=N`, or N GPUs with
`GRAD_ACCUM=1`.

---

## Assets

Both corpora reference media by paths relative to `--data-root`, e.g.
`Benchmark_tot/train/sample_222/concept_0.png`. The media themselves are ~1 GB
across three persona corpora and live in the dataset repo, under
`training_assets/`:

**https://huggingface.co/datasets/Yeongtak/Omni-Persona-Benchmark**

```bash
hf download Yeongtak/Omni-Persona-Benchmark \
  --repo-type dataset --include "training_assets/*" \
  --local-dir ./hf_data

export DATA_ROOT=$PWD/hf_data/training_assets
```

Only the 3,072 files the training corpora actually reference are published, not
the full source benchmarks. Check the wiring before launching a run:

```bash
python - <<'EOF'
import json, os, pathlib
root = pathlib.Path(os.environ["DATA_ROOT"])
need = set()
for r in json.load(open("data/rl_pairs.json")):
    for c in r["contexts"]:
        need |= {c[k] for k in ("image", "audio") if c.get(k)}
    q = r.get("query_assets") or {}
    need |= {q[k] for k in ("image", "audio") if q.get(k)}
for line in open("data/localize_verify.jsonl"):
    r = json.loads(line)
    need |= set(r.get("images") or []) | set(r.get("audios") or [])
missing = [p for p in need if not (root / p).exists()]
print(f"{len(need)} referenced, {len(missing)} missing")
EOF
```

---

## Data format

`rl_pairs.json` — one record per persona query:

| field | meaning |
|---|---|
| `contexts` | the retrieved persona memories, each with `image` / `audio` / `text` |
| `query_assets`, `query_text` | the query itself |
| `no_GT` | `true` when the queried persona is absent, i.e. the model should abstain |
| `answer_supervision` | gold answer and the asked attribute |
| `scenario`, `selected_task` | which of the four groups / eighteen tasks this is |

`localize_verify.jsonl` — one record per pre-built prompt:

| field | meaning |
|---|---|
| `messages` | chat-formatted prompt |
| `images`, `audios` | media paths, relative to `--data-root` |
| `task_type` | `image_localize` / `audio_localize` / `image_verify` / `audio_verify` |
| `no_GT` | `true` when the correct answer is "none" / "no" |
| `gt_concept_key`, `query_concept_key` | which persona is the target and which is queried |
