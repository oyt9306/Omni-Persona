#!/usr/bin/env python3
"""Export the Omni-Persona v2.2 release JSONL from the internal benchmark JSON.

The internal file carries a `source_metadata_min` block that is pure construction
bookkeeping (raw concept labels, a leftover multiple-choice audio question, and a
per-person gender map). No part of the released evaluation pipeline reads it, and the
gender map is a demographic annotation of real individuals, so it is dropped here.
Every other field is kept verbatim: `omni_persona.data` / `omni_persona.judge` /
`omni_persona.score` read `augmented_id`, `source_sample_id`, `scenario`,
`selected_task`, `query_modality`, `target_modality`, `no_GT`, `query_assets`,
`contexts[*].{image,audio,text}`, `query_text` and
`answer_supervision.{gold_answer,answerable_from_context,asked_group,asked_subfield,support_mode}`.

Output is deterministic: items are sorted by (source sample index, augmentation index)
and every JSON object is written with sorted keys, so re-running produces a
byte-identical file.

Usage:
    python tools/export_dataset.py \
        --input  /path/to/augmented_context_query_pairs_v2_2.json \
        --output dist/omni_persona_v2_2.jsonl \
        --asset-root /path/to/real_benchmark/lsd
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maintainer script: it reads the internal annotation file, which is not part of
# this repository. Both paths must be supplied explicitly.
DEFAULT_INPUT = os.environ.get("OMNI_PERSONA_ANNOTATIONS")
DEFAULT_ASSET_ROOT = os.environ.get("OMNI_PERSONA_ASSET_ROOT")

# Top-level keys removed from the release. Keep this list explicit and short:
# anything not listed here is shipped verbatim.
STRIPPED_KEYS = ("source_metadata_min",)

# scenario -> task group, frozen (see docs/DATASET_CARD.md).
GROUP: Dict[str, str] = {
    "visual_identity": "I2I",
    "voice_identity": "A2A",
    "same_modal_semantic": "T2T",
    "cross_modal_semantic_bridge": "T2Any",
}
GROUP_ORDER = ["I2I", "A2A", "T2T", "T2Any"]

_ID_RE = re.compile(r"^sample_(\d+)__aug_(\d+)$")


def sort_key(item: Dict[str, Any]) -> Tuple[int, int, str]:
    """Natural sort on `augmented_id` so sample_2 precedes sample_10."""
    aug_id = str(item.get("augmented_id", ""))
    match = _ID_RE.match(aug_id)
    if match is None:
        logger.warning("augmented_id does not match sample_<i>__aug_<j>: %r", aug_id)
        return (1 << 30, 1 << 30, aug_id)
    return (int(match.group(1)), int(match.group(2)), aug_id)


def strip_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if k not in STRIPPED_KEYS}


def load_items(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        logger.error("Benchmark file not found: %s", path)
        raise
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}")
    return data


def asset_paths(item: Dict[str, Any]) -> List[str]:
    paths = [p for p in item.get("query_assets", {}).values() if p]
    for ctx in item.get("contexts", []):
        paths.extend(ctx[k] for k in ("image", "audio") if ctx.get(k))
    return paths


def verify_assets(items: List[Dict[str, Any]], asset_root: Path) -> Tuple[int, List[str]]:
    """Return (n_referenced_assets, missing_relative_paths)."""
    referenced = sorted({p for item in items for p in asset_paths(item)})
    missing = [p for p in referenced if not (asset_root / p).exists()]
    return len(referenced), missing


def print_summary(items: List[Dict[str, Any]]) -> None:
    n_ans = sum(1 for i in items if i["answer_supervision"]["answerable_from_context"])
    n_unans = len(items) - n_ans

    print(f"\n=== Omni-Persona v2.2 export summary ===")
    print(f"items                  {len(items)}")
    print(f"source samples         {len(set(i['source_sample_id'] for i in items))}")
    print(f"answerable             {n_ans}")
    print(f"unanswerable           {n_unans}")
    print(f"fine-grained tasks     {len(set(i['selected_task'] for i in items))}")

    per_group: Dict[str, Counter] = defaultdict(Counter)
    for item in items:
        group = GROUP.get(item["scenario"], "?")
        sup = item["answer_supervision"]
        per_group[group]["n"] += 1
        if sup["answerable_from_context"]:
            per_group[group]["ans"] += 1
        else:
            per_group[group]["unans"] += 1
            per_group[group]["no_GT" if item["no_GT"] else "attr_absent"] += 1

    print("\ngroup   scenario                       total   ans  unans   (no_GT / attr_absent)")
    for group in GROUP_ORDER:
        scenario = next(s for s, g in GROUP.items() if g == group)
        c = per_group[group]
        print(
            f"{group:<7} {scenario:<28} {c['n']:>6} {c['ans']:>5} {c['unans']:>6}"
            f"   ({c['no_GT']} / {c['attr_absent']})"
        )

    print("\ngroup   task                            total   ans  unans")
    per_task: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for item in items:
        key = (GROUP.get(item["scenario"], "?"), item["selected_task"])
        per_task[key]["n"] += 1
        per_task[key]["ans" if item["answer_supervision"]["answerable_from_context"] else "unans"] += 1
    for key in sorted(per_task, key=lambda k: (GROUP_ORDER.index(k[0]), k[1])):
        c = per_task[key]
        print(f"{key[0]:<7} {key[1]:<31} {c['n']:>6} {c['ans']:>5} {c['unans']:>6}")


def export(
    input_path: Path,
    output_path: Path,
    asset_root: Optional[Path],
) -> None:
    items = load_items(input_path)
    logger.info("Loaded %d items from %s", len(items), input_path)

    items = sorted(items, key=sort_key)
    ids = [i["augmented_id"] for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate augmented_id values in the source file")

    released = [strip_item(item) for item in items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in released:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    size_mb = output_path.stat().st_size / 1024 ** 2
    print(f"[export] wrote {len(released)} lines -> {output_path} ({size_mb:.1f} MB)")
    print(f"[export] stripped top-level fields: {', '.join(STRIPPED_KEYS)}")

    print_summary(items)

    if asset_root is not None:
        n_refs, missing = verify_assets(items, asset_root)
        print(f"\nassets root            {asset_root}")
        print(f"referenced assets      {n_refs}")
        print(f"missing assets         {len(missing)}")
        for path in missing[:10]:
            print(f"  missing: {path}")
        if missing:
            raise SystemExit("[error] referenced assets are missing; fix before release")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path("dist/omni_persona_v2_2.jsonl"))
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(DEFAULT_ASSET_ROOT),
        help="Root that context/query relative asset paths resolve against.",
    )
    parser.add_argument("--skip-asset-check", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    export(args.input, args.output, None if args.skip_asset_check else args.asset_root)


if __name__ == "__main__":
    main()
