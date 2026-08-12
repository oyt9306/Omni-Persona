"""Verifiable rewards for Omni-Persona RLVR.

Three components, each scoring one capability the benchmark isolates:

  localize  the model must name the index of the context holding the queried
            persona, or the "none" index when the persona is absent
  verify    the model must answer yes/no to "are these two the same person?"
  text_qa   the model must answer from the located context, or abstain when
            the context does not contain the answer

The payoffs are deliberately asymmetric. Writing the four outcomes of a
verifiable decision as

    TP  a correct grounded answer, or a correct match
    TN  a correct abstention, or a correct rejection
    FN  a false abstention, or a missed match
    FP  an unsupported answer, or a false match

the reward is

    r(TP) = r(TN) = +1.0        r(FN) = r(FP) = -0.5

and every remaining outcome -- an incorrect grounded answer, an unparsable
response, a wrong concrete index -- scores 0.

Rewarding TP and TN equally stops the policy from inflating its apparent
grounding by answering everything, while the smaller penalty on FN/FP keeps a
single mistake from dominating a rollout group. The judge is binary: an answer
is correct or it is not, with no partial credit.

Every positive reward is gated on a degeneration check, so a repetitive
completion that happens to contain the right token earns nothing.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Callable, Optional

# Read from the environment so a sweep can vary them without touching this file;
# the defaults are the scheme reported in the paper.
R_TP = float(os.getenv("RLVR_R_TP", "1.0"))    # correct answer / correct match
R_TN = float(os.getenv("RLVR_R_TN", "1.0"))    # correct abstention / correct rejection
R_FN = float(os.getenv("RLVR_R_FN", "-0.5"))   # false abstention / missed match
R_FP = float(os.getenv("RLVR_R_FP", "-0.5"))   # unsupported answer / false match

_FINAL_ANSWER_CHARS = 300

# Matched against the final sentence only. Reasoning routinely contains
# "I cannot tell from context 0, but context 2 matches", and scanning the whole
# completion turns those into false abstentions.
_ABSTAIN_PATTERNS = [
    r"cannot determine", r"cannot be determined", r"not enough information",
    r"insufficient information", r"cannot answer", r"unable to determine",
    r"don't know from", r"do not know from", r"the provided context does not",
    r"not provided in the context", r"no information in the context",
    r"context does not contain",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


# --------------------------------------------------------------------------- #
# completion parsing
# --------------------------------------------------------------------------- #
def predict_index(completion: str) -> Optional[str]:
    """Pull the predicted context index out of a completion.

    Tries the required \\boxed{} format first, taking the LAST match because
    models also box intermediate guesses; then an "Answer: X" tail; then a bare
    digit in the final 80 characters, which is narrow enough not to catch
    context indices cited mid-reasoning.
    """
    boxed = re.findall(r"\\boxed\{(\d)\}", completion)
    if boxed:
        return boxed[-1]
    tail = completion[-_FINAL_ANSWER_CHARS:]
    m = re.search(r"\bAnswer\s*:\s*(\d)\b", tail, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d)\b", completion[-80:])
    return m.group(1) if m else None


def predict_yesno(completion: str) -> Optional[str]:
    """Pull a yes/no verdict out of a completion, preferring the final one."""
    hits = re.findall(r"\b(yes|no)\b", completion, re.IGNORECASE)
    return hits[-1].lower() if hits else None


def is_abstaining(completion: str) -> bool:
    """True when the LAST non-trivial sentence declines to answer."""
    chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", completion)
              if len(s.strip()) > 8]
    return bool(chunks) and bool(_ABSTAIN_RE.search(chunks[-1]))


# --------------------------------------------------------------------------- #
# degeneration gate
# --------------------------------------------------------------------------- #
def ngram_repetition_ratio(text: str, n: int = 4) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 2 * n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def is_degenerate(text: str) -> bool:
    """Reject completions that are repetitive rather than reasoned."""
    if not text or not text.strip():
        return True
    if len(set(text)) / len(text) < 0.05:
        return True
    if ngram_repetition_ratio(text) > 0.6:
        return True
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) >= 10]
    return any(c >= 2 for c in Counter(sentences).values())


def gate(reward: float, completion: str) -> float:
    """Positive rewards only survive a non-degenerate completion."""
    return 0.0 if is_degenerate(completion) else reward


# --------------------------------------------------------------------------- #
# reward components
# --------------------------------------------------------------------------- #
def reward_localize(completion: str, gt_index: str, n_contexts: int) -> float:
    """Score an index prediction. `n_contexts` is the index meaning "none"."""
    if not completion or not completion.strip():
        return 0.0
    pred = predict_index(completion)
    if pred is None:
        return 0.0
    if pred == gt_index:
        return gate(R_TP if gt_index != str(n_contexts) else R_TN, completion)
    # Crossing the none/context boundary is the failure the metric cares about;
    # picking the wrong concrete context is merely unrewarded.
    if (pred == str(n_contexts)) != (gt_index == str(n_contexts)):
        return gate(R_FN if gt_index != str(n_contexts) else R_FP, completion)
    return 0.0


def reward_verify(completion: str, label: str) -> float:
    """Score a same-person yes/no judgement."""
    if not completion or not completion.strip():
        return 0.0
    pred = predict_yesno(completion)
    if pred is None:
        return 0.0
    label = str(label).strip().lower()
    if label == "yes":
        return gate(R_TP if pred == "yes" else R_FN, completion)
    if label == "no":
        return gate(R_TN if pred == "no" else R_FP, completion)
    return 0.0


def reward_text_qa(
    completion: str,
    gold_answer: str,
    answerable: bool,
    judge: Optional[Callable[[str, str], str]] = None,
) -> float:
    """Score a free-form answer, or an abstention on an unanswerable item.

    `judge` maps (gold_answer, completion) -> "correct" | "incorrect". A wrong
    grounded answer is not a false positive in the abstention sense -- the model
    did commit to an answer -- so it scores 0 rather than r(FP).
    """
    if not completion or not completion.strip():
        return 0.0

    abstaining = is_abstaining(completion)

    if not answerable:
        return gate(R_TN if abstaining else R_FP, completion)

    if abstaining:
        return gate(R_FN, completion)

    if judge is None:
        raise ValueError("answerable items need a judge callable")
    return gate(R_TP, completion) if judge(gold_answer, completion) == "correct" else 0.0


REWARD_FNS = {
    "audio_localize": reward_localize,
    "image_localize": reward_localize,
    "audio_pair_verify": reward_verify,
    "image_pair_verify": reward_verify,
    "text_qa": reward_text_qa,
}
