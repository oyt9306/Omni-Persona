"""GSPO training entry point for Omni-Persona RLVR on Qwen2.5-Omni.

Reference implementation of the RLVR recipe described in the paper. It wires
three pieces together and leaves everything else to TRL:

  dataset.py  samples the 30:30:40 localize / verify / text_qa mixture
  reward.py   scores a completion according to which task it was drawn from
  GRPOConfig  run with importance_sampling_level="sequence", which is GSPO

The text_qa reward needs an LLM judge. Point --judge-base-url at any
OpenAI-compatible endpoint; the run aborts up front if it does not answer,
because a silently dead judge makes every text_qa reward 0.0 and the run looks
like it is merely learning slowly.

Only the Thinker is trained. Qwen2.5-Omni's Talker is a speech decoder that
takes no gradient from a text-only objective, and loading it wastes memory.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import torch

from omni_persona_rlvr.dataset import OmniPersonaRLVRDataset
from omni_persona_rlvr.reward import (
    reward_localize,
    reward_text_qa,
    reward_verify,
)


def build_judge(base_url: str, model: str):
    """Return (gold, completion) -> 'correct' | 'incorrect'."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
    system = (
        "You are evaluating whether a model's final answer is correct.\n"
        "The model sees several persona memories and must identify the person the "
        "query refers to, then answer about that person.\n"
        "Reply with ONLY 'correct' or 'incorrect'."
    )

    def judge(gold: str, completion: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",
                 "content": f"Gold answer: {gold}\nModel answer: {completion}\n\nVerdict?"},
            ],
            max_completion_tokens=8,
        )
        verdict = (resp.choices[0].message.content or "").strip().lower()
        return "correct" if verdict.startswith("correct") else "incorrect"

    return judge


def preflight_judge(judge) -> None:
    try:
        judge("Seoul", "The answer is Seoul.")
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal
        sys.exit(f"FATAL judge unreachable: {exc}")


def make_reward_fn(judge):
    """One TRL reward callable that dispatches on the sampled task type."""

    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        task_types = kwargs["task_type"]
        out: List[float] = []
        for i, completion in enumerate(completions):
            task = task_types[i]
            if task.endswith("localize"):
                out.append(reward_localize(
                    completion, kwargs["gt_index"][i], kwargs["n_contexts"][i]))
            elif task.endswith("verify"):
                out.append(reward_verify(completion, kwargs["verify_label"][i]))
            else:
                out.append(reward_text_qa(
                    completion, kwargs["gold_answer"][i], kwargs["answerable"][i],
                    judge=judge))
        return out

    return reward_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    p.add_argument("--rl-json", required=True)
    p.add_argument("--lv-jsonl", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--judge-base-url", default="http://localhost:8091/v1")
    p.add_argument("--judge-model", default="gpt-5.4-mini")
    p.add_argument("--localize-weight", type=float, default=0.30)
    p.add_argument("--verify-weight", type=float, default=0.30)
    p.add_argument("--textqa-weight", type=float, default=0.40)
    p.add_argument("--unans-ratio", type=float, default=0.6)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--max-prompt-length", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.1)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--learning-rate", type=float, default=1e-6)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--deepspeed", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    from trl import GRPOConfig, GRPOTrainer

    judge = build_judge(args.judge_base_url, args.judge_model)
    preflight_judge(judge)

    dataset = OmniPersonaRLVRDataset(
        rl_json=args.rl_json,
        lv_jsonl=args.lv_jsonl,
        data_root=args.data_root,
        localize_weight=args.localize_weight,
        verify_weight=args.verify_weight,
        textqa_weight=args.textqa_weight,
        unans_ratio=args.unans_ratio,
        seed=args.seed,
    )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    config = GRPOConfig(
        output_dir=args.output_dir,
        # GSPO is GRPO with sequence-level importance sampling. On this task the
        # token-level variant lets a short "I cannot determine" shortcut dominate.
        importance_sampling_level="sequence",
        beta=args.beta,
        epsilon=0.2,
        epsilon_high=0.2,
        num_iterations=1,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        seed=args.seed,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=make_reward_fn(judge),
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
