"""Run a model over the benchmark against an OpenAI-compatible server.

    python -m omni_persona.infer \
        --input data/augmented_context_query_pairs_v2_2.json \
        --asset-root data/lsd \
        --base-url http://localhost:8001/v1 --model m \
        --output results/qwen3b_base/predictions.jsonl

Defaults (frozen): temperature=0.0, max_tokens=256.
Re-running with the same --output resumes: already-written items are skipped.
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

from .data import build_multimodal_messages, load_benchmark, stratified_subset

logger = logging.getLogger(__name__)


class InferenceClient:
    """Minimal chat-completions client with retry/backoff."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: int = 180,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def chat(self, messages: List[Dict], pre_delay: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if pre_delay > 0:
            time.sleep(pre_delay)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code in {400, 500}:
                    # Non-retryable:
                    # 400 — context length exceeded
                    # 500 — vLLM mm_receiver_cache: retrying the same multimodal
                    #       content re-uses a consumed mm_hash and fails identically.
                    raise ValueError(f"{resp.status_code} Error: {resp.text[:200]}")
                if resp.status_code in {429, 502, 503, 504}:
                    raise requests.exceptions.HTTPError(
                        f"retryable {resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except ValueError:
                raise  # don't retry 400s
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last_err = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff ** attempt + random.uniform(0, 0.5))
        raise last_err  # type: ignore[misc]


def _record(item: Dict[str, Any], model: str, prediction: str) -> Dict[str, Any]:
    sup = item.get("answer_supervision", {})
    return {
        "augmented_id": item.get("augmented_id"),
        "source_sample_id": item.get("source_sample_id"),
        "model_name": model,
        "scenario": item.get("scenario"),
        "selected_task": item.get("selected_task"),
        "query_modality": item.get("query_modality"),
        "target_modality": item.get("target_modality"),
        "no_GT": item.get("no_GT"),
        "query_text": item.get("query_text"),
        "model_prediction": prediction,
        "gold_answer": sup.get("gold_answer"),
        "answerable_from_context": sup.get("answerable_from_context"),
        "asked_subfield": sup.get("asked_subfield"),
        "asked_group": sup.get("asked_group"),
        "support_mode": sup.get("support_mode"),
    }


def _done_ids(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["augmented_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Omni-Persona v2.2 inference")
    ap.add_argument("--input", required=True, help="augmented_context_query_pairs_v2_2.json(l)")
    ap.add_argument("--asset-root", required=True, help="dir with sample_*/concept_*.png|wav")
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:8001/v1")
    ap.add_argument("--model", required=True, help="model name as registered by the server")
    ap.add_argument("--output", required=True, help="predictions .jsonl path")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--num-workers", type=int, default=1,
                    help="1 reproduces the published runs; >1 is faster but can trip "
                         "vLLM's multimodal receiver cache")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--request-delay", type=float, default=0.5,
                    help="sleep before each request (lets the vLLM mm cache settle)")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate a stratified subset of N items (0 = full 750)")
    ap.add_argument("--skip-audio", action="store_true",
                    help="drop audio blocks (for text/image-only models)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="cap total image blocks per prompt (e.g. 4 for Phi-4-multimodal)")
    args = ap.parse_args()

    asset_root = Path(args.asset_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_benchmark(args.input)
    if args.limit:
        items = stratified_subset(items, args.limit)
    done = _done_ids(out_path)
    pending = [it for it in items if it.get("augmented_id") not in done]

    logger.info("total=%d done=%d pending=%d", len(items), len(done), len(pending))
    logger.info("model=%s  asset_root=%s  output=%s", args.model, asset_root, out_path)

    client = InferenceClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    write_lock = threading.Lock()
    errors: List[str] = []

    def process(item: Dict[str, Any]) -> None:
        aug_id = item.get("augmented_id", "?")
        try:
            messages = build_multimodal_messages(
                item, asset_root, skip_audio=args.skip_audio, max_images=args.max_images
            )
            prediction = client.chat(messages, pre_delay=args.request_delay)
        except ValueError as exc:
            # 400 Bad Request (e.g. context length exceeded): record a placeholder
            # so the item is present and counted rather than silently missing.
            prediction = "[SKIP] " + str(exc)[:120]
            logger.warning("skip %s: %s", aug_id, exc)
        except Exception as exc:  # noqa: BLE001
            with write_lock:
                errors.append(str(aug_id))
            logger.error("%s: %s", aug_id, exc)
            return
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(_record(item, client.model, prediction), ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(process, it) for it in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
            pass

    logger.info("wrote %d predictions → %s", len(pending) - len(errors), out_path)
    if errors:
        logger.warning("%d failed: %s", len(errors), errors[:10])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
