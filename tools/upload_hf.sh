#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# upload_hf.sh — publish the Omni-Persona v2.2 release to the Hugging Face Hub.
#
#   DRY RUN BY DEFAULT. Nothing is written to the Hub unless you pass --yes.
#
# What it pushes
#   dataset  Yeongtak/Omni-Persona-Benchmark
#              omni_persona_v2_2.jsonl        (from tools/export_dataset.py)
#              README.md                      (from docs/DATASET_CARD.md)
#              assets/lsd/**                  (~1.2 GB, 2500 files)
#              lsd_metadata.jsonl, gender_cache.json
#   model    Yeongtak/Qwen2.5-Omni-3B-Omni-Persona-RLVR   (12 GB)
#   model    Yeongtak/Qwen2.5-Omni-7B-Omni-Persona-RLVR   (21 GB)
#
# Usage
#   bash tools/upload_hf.sh                        # dry run, everything
#   bash tools/upload_hf.sh --only dataset         # dry run, dataset only
#   bash tools/upload_hf.sh --only models --yes    # REAL upload of both models
#   bash tools/upload_hf.sh --public --yes         # create repos public (default: private)
#
# Prerequisites
#   pip install -U "huggingface_hub[cli,hf_transfer]"
#   export HF_TOKEN=hf_...            # a WRITE token
#   python tools/export_dataset.py --output dist/omni_persona_v2_2.jsonl
#
# READ THIS FIRST
#     Both target namespaces below are a personal handle. If the paper is under
#     double-blind review, publishing under them de-anonymizes the submission.
#   * Repos are created PRIVATE by default on purpose. Flip to public only after the
#     checklist is signed off.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- configuration ---------------------------------------------------------
DATASET_REPO="${DATASET_REPO:-Yeongtak/Omni-Persona-Benchmark}"
MODEL_REPO_3B="${MODEL_REPO_3B:-Yeongtak/Qwen2.5-Omni-3B-Omni-Persona-RLVR}"
MODEL_REPO_7B="${MODEL_REPO_7B:-Yeongtak/Qwen2.5-Omni-7B-Omni-Persona-RLVR}"

DATASET_JSONL="${DATASET_JSONL:-$REPO_ROOT/dist/omni_persona_v2_2.jsonl}"
DATASET_CARD="${DATASET_CARD:-$REPO_ROOT/docs/DATASET_CARD.md}"
MODEL_CARD="${MODEL_CARD:-$REPO_ROOT/docs/MODEL_CARD.md}"

# Maintainer script: these point into the author's working tree and have no
# sensible default here. Export them before running.
BENCH_ROOT="${BENCH_ROOT:?set BENCH_ROOT to the directory containing lsd/}"
ASSET_DIR="$BENCH_ROOT/lsd"

CKPT_3B="${CKPT_3B:?set CKPT_3B to the merged 3B checkpoint directory}"
CKPT_7B="${CKPT_7B:?set CKPT_7B to the merged 7B checkpoint directory}"

# --- argument parsing ------------------------------------------------------
CONFIRM=0
ONLY="all"
PRIVATE="--private"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)    CONFIRM=1; shift ;;
    --only)   ONLY="${2:-}"; shift 2 ;;
    --public) PRIVATE=""; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$ONLY" in
  all|dataset|models) ;;
  *) echo "--only must be one of: all, dataset, models" >&2; exit 2 ;;
esac

run() {
  # Echo the command; execute it only when --yes was given.
  if [[ "$CONFIRM" -eq 1 ]]; then
    echo "+ $*"
    "$@"
  else
    echo "  [dry-run] $*"
  fi
}

banner() { echo; echo "=== $* ==="; }

# --- preflight -------------------------------------------------------------
banner "preflight"

fail=0
need_file() { [[ -f "$1" ]] || { echo "  MISSING file: $1"; fail=1; }; }
need_dir()  { [[ -d "$1" ]] || { echo "  MISSING dir:  $1"; fail=1; }; }

if [[ "$ONLY" == "all" || "$ONLY" == "dataset" ]]; then
  need_file "$DATASET_JSONL"
  need_file "$DATASET_CARD"
  need_dir  "$ASSET_DIR"
fi
if [[ "$ONLY" == "all" || "$ONLY" == "models" ]]; then
  need_file "$MODEL_CARD"
  need_dir  "$CKPT_3B"
  need_dir  "$CKPT_7B"
fi
[[ "$fail" -eq 0 ]] || { echo "preflight failed"; exit 1; }

if [[ -f "$DATASET_JSONL" ]]; then
  n_lines=$(wc -l < "$DATASET_JSONL")
  echo "  dataset jsonl : $DATASET_JSONL ($n_lines lines)"
  if [[ "$n_lines" -ne 750 ]]; then
    echo "  ERROR: expected 750 items, found $n_lines. Re-run tools/export_dataset.py." >&2
    exit 1
  fi
fi
if [[ -d "$ASSET_DIR" ]]; then
  echo "  assets        : $ASSET_DIR ($(find "$ASSET_DIR" -type f | wc -l) files, $(du -sh "$ASSET_DIR" | cut -f1))"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ "$CONFIRM" -eq 1 ]]; then
    echo "  ERROR: HF_TOKEN is not set (needs a WRITE token)." >&2
    exit 1
  fi
  echo "  HF_TOKEN      : (unset — fine for a dry run)"
else
  echo "  HF_TOKEN      : set (${#HF_TOKEN} chars)"
fi

if [[ "$CONFIRM" -ne 1 ]]; then
  cat <<'EOF'

  ############################################################
  #  DRY RUN — nothing will be uploaded.                     #
  #  Re-run with --yes to actually publish.                  #
  ############################################################
EOF
else
  cat <<EOF

  !!! LIVE UPLOAD !!!
      dataset : $DATASET_REPO
      models  : $MODEL_REPO_3B
                $MODEL_REPO_7B
      privacy : ${PRIVATE:---public}
EOF
  read -r -p "  Type the word PUBLISH to continue: " ack
  [[ "$ack" == "PUBLISH" ]] || { echo "  aborted."; exit 1; }
fi

# Faster large-file transfer; harmless if hf_transfer is absent.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# --- dataset ---------------------------------------------------------------
if [[ "$ONLY" == "all" || "$ONLY" == "dataset" ]]; then
  banner "dataset -> $DATASET_REPO"

  run hf repo create "$DATASET_REPO" --repo-type dataset --exist-ok $PRIVATE

  # The dataset card must land as README.md at the repo root for the Hub to render it.
  run hf upload "$DATASET_REPO" "$DATASET_CARD" README.md \
      --repo-type dataset --commit-message "Add v2.2 dataset card"

  run hf upload "$DATASET_REPO" "$DATASET_JSONL" omni_persona_v2_2.jsonl \
      --repo-type dataset --commit-message "Add Omni-Persona v2.2 annotations (750 items)"

  # ~1.2 GB / 2500 small files: one folder upload, not 2500 file uploads.
  run hf upload "$DATASET_REPO" "$ASSET_DIR" assets/lsd \
      --repo-type dataset --exclude "*.DS_Store" --exclude "__pycache__/*" \
      --commit-message "Add benchmark assets (images + audio)"

  run hf upload "$DATASET_REPO" "$BENCH_ROOT/lsd_metadata.jsonl" lsd_metadata.jsonl \
      --repo-type dataset --commit-message "Add construction metadata"

  run hf upload "$DATASET_REPO" "$BENCH_ROOT/gender_cache.json" gender_cache.json \
      --repo-type dataset --commit-message "Add distractor gender cache"

  echo "  -> https://huggingface.co/datasets/$DATASET_REPO"
fi

# --- models ----------------------------------------------------------------
upload_model() {
  local repo="$1" ckpt="$2" label="$3"
  banner "model $label -> $repo"
  echo "  source: $ckpt ($(du -sh "$ckpt" | cut -f1))"

  run hf repo create "$repo" --repo-type model --exist-ok $PRIVATE

  run hf upload "$repo" "$MODEL_CARD" README.md \
      --repo-type model --commit-message "Add model card"

  # Weights + tokenizer + preprocessor configs in one shot.
  run hf upload "$repo" "$ckpt" . \
      --repo-type model --exclude "*.pt" --exclude "optimizer*" --exclude "__pycache__/*" --exclude ".git/*" \
      --commit-message "Add merged RLVR checkpoint (v20 30:30:40 GSPO, step 100)"

  echo "  -> https://huggingface.co/$repo"
  echo "  NOTE: docs/MODEL_CARD.md covers BOTH checkpoints. After upload, trim the pushed"
  echo "        README.md down to the $label rows and fix the base_model frontmatter."
}

if [[ "$ONLY" == "all" || "$ONLY" == "models" ]]; then
  upload_model "$MODEL_REPO_3B" "$CKPT_3B" "3B"
  upload_model "$MODEL_REPO_7B" "$CKPT_7B" "7B"
fi

banner "done"
if [[ "$CONFIRM" -ne 1 ]]; then
  echo "This was a DRY RUN. No data left this machine."
fi
