from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, TypeVar

import torch

from cs336_alignment.rl.fixtures import extract_gsm8k_answer, gsm8k_reward_fn, make_rollouts
from cs336_alignment.rl.grpo import grpo_train_step
from cs336_alignment.rl.models import tiny_byte_tokenizer, tiny_train_model

T = TypeVar("T")
logger = logging.getLogger(__name__)


def progress(iterable: Iterable[T], **kwargs) -> Iterable[T]:
  try:
    from tqdm.auto import tqdm
  except ImportError:
    return iterable
  return tqdm(iterable, **kwargs)


def load_gsm8k_examples(path: Path, limit: int) -> list[dict[str, str]]:
  examples: list[dict[str, str]] = []
  logger.info("Loading %d GSM8K examples from %s", limit, path)
  with path.open() as f:
    for line in f:
      raw = json.loads(line)
      examples.append(
        {
          "question": raw["question"],
          "answer": extract_gsm8k_answer(raw["answer"]),
        }
      )
      if len(examples) == limit:
        break
  logger.info("Loaded %d examples", len(examples))
  return examples


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Local smoke test for grpo_train_step on GSM8K.")
  parser.add_argument("--data-path", type=Path, default=Path("data/gsm8k/train.jsonl"))
  parser.add_argument("--num-prompts", type=int, default=2)
  parser.add_argument("--group-size", type=int, default=2)
  parser.add_argument("--steps", type=int, default=1)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
  parser.add_argument("--max-grad-norm", type=float, default=1.0)
  parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
  )
  torch.manual_seed(0)
  logger.info("Starting local GRPO smoke test")
  logger.info(
    "Config: num_prompts=%d group_size=%d rollout_batch=%d steps=%d lr=%g grad_accum=%d",
    args.num_prompts,
    args.group_size,
    args.num_prompts * args.group_size,
    args.steps,
    args.lr,
    args.gradient_accumulation_steps,
  )

  examples = load_gsm8k_examples(args.data_path, args.num_prompts)
  logger.info("Building rollout batch with group_size=%d", args.group_size)
  prompts, outputs, ground_truths = make_rollouts(examples, args.group_size)
  logger.info(
    "Built rollout batch: prompts=%d outputs=%d ground_truths=%d",
    len(prompts),
    len(outputs),
    len(ground_truths),
  )
  logger.info("Creating byte-level tokenizer with 256 byte tokens + 2 specials")
  tokenizer = tiny_byte_tokenizer()
  logger.info(
    "Tokenizer ready: vocab_size=%d eos_token_id=%d pad_token_id=%d ",
    len(tokenizer),
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
  )
  logger.info("Creating tiny_train_model on CPU")
  model = tiny_train_model(tokenizer, device=torch.device("cpu"))
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
  num_params = sum(param.numel() for param in model.parameters())
  logger.info("Model ready on CPU with %d parameters and context length %d", num_params, model.config.n_positions)
  logger.info("Running GRPO train steps")

  for step in progress(range(args.steps), desc="training", unit="step"):
    logger.debug("Starting step %d", step)
    result = grpo_train_step(
      model=model,
      tokenizer=tokenizer,
      optimizer=optimizer,
      reward_fn=gsm8k_reward_fn,
      prompt_strs=prompts,
      output_strs=outputs,
      ground_truths=ground_truths,
      group_size=args.group_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      max_grad_norm=args.max_grad_norm,
    )
    message = (
      "step={step} loss={loss:.6f} reward_mean={reward_mean:.3f} "
      "reward_min={reward_min:.3f} reward_max={reward_max:.3f} "
      "advantage_mean={advantage_mean:.3f} advantage_std={advantage_std:.3f} "
      "log_prob_shape={log_prob_shape}".format(
        step=step,
        loss=result.loss.item(),
        reward_mean=result.rewards.float().mean().item(),
        reward_min=result.rewards.float().min().item(),
        reward_max=result.rewards.float().max().item(),
        advantage_mean=result.advantages.float().mean().item(),
        advantage_std=result.advantages.float().std(unbiased=False).item(),
        log_prob_shape=tuple(result.log_probs.shape),
      )
    )
    logger.info(message)

  logger.info("Done")


if __name__ == "__main__":
  main()
