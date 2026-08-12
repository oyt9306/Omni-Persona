"""LLM-as-a-judge verdicts + keyword abstention detection.

    python -m omni_persona.judge \
        --predictions results/qwen3b_base/predictions.jsonl \
        --judge-base-url http://localhost:8091/v1 --judge-model gpt-5.4-mini \
        --output results/qwen3b_base/judge_results.jsonl

FROZEN CONTRACTS: ``is_abstain`` keyword list, ``JUDGE_SYSTEM`` /
``JUDGE_USER_TEMPLATE``, and the strict verdict parsing (only ``CORRECT``
counts as correct) are copied verbatim from the research pipeline.
"""

import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

ABSTAIN_ANSWER = "I cannot determine that from the provided context."

JUDGE_SYSTEM = """You are an evaluation judge for a question-answering benchmark.
You will be given a gold answer and a model prediction. Decide if the prediction is correct.

Rules:
- For factual answers (non-abstain): the prediction is CORRECT if its final answer conveys the same core information as the gold answer, even with different wording. Minor paraphrasing is fine.
  - The prediction is also CORRECT if the gold answer is explicitly contained within the prediction as a clear, stated fact (not merely mentioned in passing about a different person).
  - The model receives multiple memory contexts and must identify the correct person before answering. If the model lists multiple conflicting answers across different people, or mentions the gold answer only in passing while attributing a different answer as its conclusion, it is WRONG.
  - If the model hedges without committing to a final answer, it is WRONG.
- For abstain gold answers ("I cannot determine..."): the prediction is CORRECT only if it also expresses inability to answer (abstains). Any concrete answer is WRONG.
- Output exactly one word: CORRECT or WRONG. No explanation."""

JUDGE_USER_TEMPLATE = """Gold answer: {gold}
Model prediction: {pred}

Verdict:"""

# FROZEN: the abstention keyword list. Do not add, remove or reorder entries.
ABSTAIN_KEYWORDS = [
    "cannot determine", "cannot be determined",
    "not enough information", "insufficient information",
    "cannot answer", "unable to determine",
    "don't know from", "do not know from",
    "the provided context does not",
    "not provided in the context",
    "no information in the context",
    "context does not contain",
    "cannot identify",
    "i cannot tell",
]


def is_abstain(text: Optional[str]) -> bool:
    """True if the prediction expresses inability to answer (keyword match)."""
    if not text:
        return False
    norm = " ".join(text.strip().lower().split())
    return any(kw in norm for kw in ABSTAIN_KEYWORDS)


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer (diagnostic only)."""
    def _tok(s: str):
        return set(s.lower().split())
    p_tok, g_tok = _tok(pred), _tok(gold)
    if not g_tok:
        return 0.0
    common = p_tok & g_tok
    if not common:
        return 0.0
    prec = len(common) / len(p_tok) if p_tok else 0.0
    rec = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


class JudgeClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def judge(self, gold: str, pred: str) -> str:
        """Returns 'CORRECT' or 'WRONG'."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_completion_tokens": 64,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(gold=gold, pred=pred)},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.exceptions.HTTPError(
                        f"retryable {resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                verdict = resp.json()["choices"][0]["message"]["content"].strip().upper()
                return "CORRECT" if "CORRECT" in verdict else "WRONG"
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt + random.uniform(0, 0.3))
        raise last_err  # type: ignore[misc]


def judge_item(row: Dict[str, Any], client: JudgeClient) -> Dict[str, Any]:
    """Judge one prediction row.

    Unanswerable items never consult the judge: correctness is decided purely by
    ``is_abstain``. Answerable items keep the raw judge verdict; the abstention
    guard is applied later, in score.py.
    """
    gold = row.get("gold_answer", "") or ""
    pred = row.get("model_prediction", "") or ""
    answerable = bool(row.get("answerable_from_context"))

    verdict = client.judge(gold=gold, pred=pred)
    correct = verdict == "CORRECT"
    if not answerable:
        correct = is_abstain(pred)
        verdict = "CORRECT" if correct else "WRONG"

    return {
        "augmented_id": row.get("augmented_id"),
        "source_sample_id": row.get("source_sample_id"),
        "model_name": row.get("model_name"),
        "scenario": row.get("scenario"),
        "selected_task": row.get("selected_task"),
        "query_modality": row.get("query_modality"),
        "target_modality": row.get("target_modality"),
        "asked_subfield": row.get("asked_subfield"),
        "asked_group": row.get("asked_group"),
        "support_mode": row.get("support_mode"),
        "no_GT": row.get("no_GT"),
        "answerable_from_context": answerable,
        "gold_answer": gold,
        "model_prediction": pred,
        "judge_verdict": verdict,
        "correct": correct,
        "abstained": is_abstain(pred),
        "token_f1": token_f1(pred, gold) if answerable else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Omni-Persona v2.2 LLM judge")
    ap.add_argument("--predictions", required=True, help="predictions .jsonl from infer.py")
    ap.add_argument("--output", required=True, help="judge_results .jsonl path")
    ap.add_argument("--judge-base-url", default="http://localhost:8091/v1")
    ap.add_argument("--judge-model", default="gpt-5.4-mini")
    ap.add_argument("--judge-api-key", default="EMPTY")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # judging is cheap and must not mix runs

    with open(args.predictions, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    logger.info("%d predictions  judge=%s → %s", len(rows), args.judge_model, out_path)

    client = JudgeClient(
        base_url=args.judge_base_url,
        model=args.judge_model,
        api_key=args.judge_api_key,
        timeout=args.timeout,
    )

    write_lock = threading.Lock()
    errors: List[str] = []

    def process(row: Dict[str, Any]) -> None:
        try:
            result = judge_item(row, client)
        except Exception as exc:  # noqa: BLE001
            with write_lock:
                errors.append(str(row.get("augmented_id", "?")))
            logger.error("%s: %s", row.get("augmented_id"), exc)
            return
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(process, r) for r in rows]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            pass

    logger.info("judge_results → %s", out_path)
    if errors:
        logger.warning("%d errors: %s", len(errors), errors[:10])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
