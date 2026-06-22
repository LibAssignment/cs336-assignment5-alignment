from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, TypeVar

import torch

from cs336_alignment.rl.grpo import grpo_train_step
from cs336_alignment.rl.models import tiny_byte_tokenizer, tiny_train_model
from cs336_alignment.rl.prompts import (
  PROMPT_KINDS,
  extract_gsm8k_answer,
  get_prompt,
  make_prompt_rollouts,
  make_vllm_rollouts,
)

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


def parse_optimizer_params(raw_params: str) -> dict:
  try:
    params = json.loads(raw_params)
  except json.JSONDecodeError as error:
    raise argparse.ArgumentTypeError(f"Invalid optimizer params JSON: {error}") from error
  if not isinstance(params, dict):
    raise argparse.ArgumentTypeError("--optimizer-params must decode to a JSON object")
  if "betas" in params:
    params["betas"] = tuple(params["betas"])
  return params


def inference_base_url(inference: str) -> str | None:
  if inference == "smoke":
    return None
  if inference == "vllm":
    return os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
  if inference.startswith("http://") or inference.startswith("https://"):
    return inference
  raise argparse.ArgumentTypeError("--inference must be 'smoke', 'vllm', or an http(s) URL")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Local smoke test for grpo_train_step on GSM8K.")
  parser.add_argument("--data-path", type=Path, default=Path("data/gsm8k/train.jsonl"))
  parser.add_argument("--prompt", choices=PROMPT_KINDS, default="question_only")
  parser.add_argument("--inference", type=inference_base_url, default=None)
  parser.add_argument("--vllm-model", default="")
  parser.add_argument("--inference-batch-size", type=int, default=5)
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--max-tokens", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--num-prompts", type=int, default=2)
  parser.add_argument("--group-size", type=int, default=2)
  parser.add_argument("--steps", type=int, default=1)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--weight-decay", type=float, default=0.0)
  parser.add_argument(
    "--optimizer-params",
    type=parse_optimizer_params,
    default=None,
    help='Extra optimizer kwargs as JSON, e.g. \'{"betas":[0.9,0.95]}\' for AdamW.',
  )
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
  device = torch.device(args.device)
  logger.info("Starting local GRPO smoke test")
  logger.info(
    "Config: prompt=%s inference=%s num_prompts=%d group_size=%d rollout_batch=%d steps=%d device=%s optimizer=%s lr=%g weight_decay=%g grad_accum=%d",
    args.prompt,
    args.inference or "smoke",
    args.num_prompts,
    args.group_size,
    args.num_prompts * args.group_size,
    args.steps,
    device,
    args.optimizer,
    args.lr,
    args.weight_decay,
    args.gradient_accumulation_steps,
  )

  examples = load_gsm8k_examples(args.data_path, args.num_prompts)
  logger.info("Building rollout batch with group_size=%d", args.group_size)
  prompt = get_prompt(args.prompt)
  if args.inference is None:
    prompts, outputs, ground_truths = make_prompt_rollouts(prompt, examples, args.group_size)
  else:
    logger.info("Generating rollouts from inference endpoint: %s", args.inference)
    prompts, outputs, ground_truths = make_vllm_rollouts(
      args.inference,
      args.vllm_model,
      prompt,
      examples,
      args.group_size,
      args.temperature,
      args.max_tokens,
      args.seed,
      args.inference_batch_size,
    )
  logger.info(
    "Built rollout batch: prompt=%s prompts=%d outputs=%d ground_truths=%d reward_fn=%s",
    args.prompt,
    len(prompts),
    len(outputs),
    len(ground_truths),
    prompt.reward_fn.__name__,
  )
  logger.info("Creating byte-level tokenizer with 256 byte tokens + 2 specials")
  tokenizer = tiny_byte_tokenizer()
  logger.info(
    "Tokenizer ready: vocab_size=%d eos_token_id=%d pad_token_id=%d ",
    len(tokenizer),
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
  )
  logger.info("Creating tiny_train_model on %s", device)
  model = tiny_train_model(tokenizer, device=device)
  optimizer_kwargs = {
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    **(args.optimizer_params or {}),
  }
  if args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
  else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")
  num_params = sum(param.numel() for param in model.parameters())
  logger.info("Model ready on %s with %d parameters and context length %d", device, num_params, model.config.n_positions)
  logger.info("Optimizer ready: %s params=%s", optimizer.__class__.__name__, optimizer_kwargs)
  logger.info("Running GRPO train steps")

  for step in progress(range(args.steps), desc="training", unit="step"):
    logger.debug("Starting step %d", step)
    result = grpo_train_step(
      model=model,
      tokenizer=tokenizer,
      optimizer=optimizer,
      reward_fn=prompt.reward_fn,
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
