"""Task-mixture dataset for Omni-Persona RLVR.

Two corpora feed the run:

  rl_pairs.json          the persona query/answer pairs, used for text_qa
  localize_verify.jsonl  pre-built localize and verify prompts

Each __getitem__ draws a task type from the configured mixture and returns a
prompt built from the corresponding corpus, so the mixture is a sampling policy
rather than a fixed concatenation. The paper's configuration is

    localize 30% : verify 30% : text_qa 40%

with the text_qa pool held at a 1 : 0.6 answerable-to-unanswerable ratio, i.e.
roughly 37.5% of text_qa draws are absent-persona items.

Asset paths in both corpora are relative to --data_root.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

TASK_NAMES = [
    "audio_localize",
    "image_localize",
    "text_qa",
    "image_pair_verify",
    "audio_pair_verify",
]


def build_task_weights(localize: float, verify: float, text_qa: float) -> List[float]:
    """Expand three group weights over the five concrete task types."""
    total = localize + verify + text_qa
    if total <= 0:
        raise ValueError("at least one group weight must be positive")
    return [
        localize / 2.0,      # audio_localize
        localize / 2.0,      # image_localize
        text_qa,             # text_qa
        verify / 2.0,        # image_pair_verify
        verify / 2.0,        # audio_pair_verify
    ]


class OmniPersonaRLVRDataset:
    """Samples a task type per item and returns a chat-formatted prompt.

    This is the skeleton the released checkpoints were trained with. The prompt
    builders below are intentionally thin: they show the message shape the
    Qwen2.5-Omni processor expects, and the fields each reward component needs
    back, so a different backbone can be swapped in by replacing them.
    """

    def __init__(
        self,
        rl_json: str,
        lv_jsonl: str,
        data_root: str,
        localize_weight: float = 0.30,
        verify_weight: float = 0.30,
        textqa_weight: float = 0.40,
        unans_ratio: float = 0.6,
        n_contexts: int = 3,
        seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.n_contexts = n_contexts
        self.weights = build_task_weights(localize_weight, verify_weight, textqa_weight)
        self.rng = random.Random(seed)

        pairs = json.loads(Path(rl_json).read_text(encoding="utf-8"))
        answerable = [r for r in pairs if not r.get("no_GT")]
        unanswerable = [r for r in pairs if r.get("no_GT")]
        # Hold the answerable:unanswerable balance at 1:unans_ratio by trimming
        # whichever side is over-represented, rather than oversampling.
        keep_unans = min(len(unanswerable), int(len(answerable) * unans_ratio))
        self.rng.shuffle(unanswerable)
        self.text_qa_pool = answerable + unanswerable[:keep_unans]
        self.rng.shuffle(self.text_qa_pool)

        lv = [json.loads(l) for l in Path(lv_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.lv_pools: Dict[str, List[dict]] = {}
        for row in lv:
            self.lv_pools.setdefault(row["task_type"], []).append(row)
        # verify rows are stored under image_verify / audio_verify
        self.lv_pools.setdefault("image_pair_verify", self.lv_pools.get("image_verify", []))
        self.lv_pools.setdefault("audio_pair_verify", self.lv_pools.get("audio_verify", []))

        print(
            f"[dataset] text_qa pool={len(self.text_qa_pool)} "
            f"({len(answerable)} answerable / {keep_unans} unanswerable) | "
            + " ".join(f"{k}={len(v)}" for k, v in sorted(self.lv_pools.items()))
        )

    def __len__(self) -> int:
        return len(self.text_qa_pool)

    def _resolve(self, rel: str) -> str:
        return str(self.data_root / rel)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.rng.choices(TASK_NAMES, weights=self.weights, k=1)[0]
        if task == "text_qa":
            return self._text_qa_item(self.text_qa_pool[idx % len(self.text_qa_pool)])
        pool = self.lv_pools.get(task) or []
        if not pool:
            return self._text_qa_item(self.text_qa_pool[idx % len(self.text_qa_pool)])
        return self._prebuilt_item(pool[self.rng.randrange(len(pool))], task)

    # ----------------------------------------------------------------- items
    def _text_qa_item(self, row: dict) -> Dict[str, Any]:
        """Query + retrieved contexts, answered from the matching context."""
        content: List[dict] = []
        for ctx in row["contexts"][: self.n_contexts]:
            if ctx.get("image"):
                content.append({"type": "image", "image": self._resolve(ctx["image"])})
            if ctx.get("audio"):
                content.append({"type": "audio", "audio": self._resolve(ctx["audio"])})
            if ctx.get("text"):
                content.append({"type": "text", "text": ctx["text"]})
        q = row.get("query_assets") or {}
        if q.get("image"):
            content.append({"type": "image", "image": self._resolve(q["image"])})
        if q.get("audio"):
            content.append({"type": "audio", "audio": self._resolve(q["audio"])})
        content.append({"type": "text", "text": row["query_text"]})

        sup = row.get("answer_supervision") or {}
        return {
            "task_type": "text_qa",
            "prompt": [{"role": "user", "content": content}],
            "gold_answer": sup.get("gold_answer", ""),
            "answerable": not row.get("no_GT", False),
        }

    def _prebuilt_item(self, row: dict, task: str) -> Dict[str, Any]:
        """Localize / verify prompts already carry their own message list."""
        msgs = []
        for m in row["messages"]:
            if isinstance(m.get("content"), str):
                msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
            else:
                msgs.append(m)
        media = [self._resolve(p) for p in (row.get("images") or [])] + \
                [self._resolve(p) for p in (row.get("audios") or [])]
        return {
            "task_type": task,
            "prompt": msgs,
            "media": media,
            # localize: index of the matching context, or n_contexts for "none"
            "gt_index": row.get("gt_index", str(self.n_contexts) if row.get("no_GT") else None),
            "n_contexts": self.n_contexts,
            # verify: "yes" when the pair is the same person
            "verify_label": "no" if row.get("no_GT") else "yes",
        }
