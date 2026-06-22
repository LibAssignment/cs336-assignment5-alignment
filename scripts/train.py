from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, TypeVar

import torch

from cs336_alignment.rl.grpo import grpo_train_step
from cs336_alignment.rl.config import TrainConfig, WandbConfig, parse_train_config
from cs336_alignment.rl.models import tiny_byte_tokenizer, tiny_train_model
from cs336_alignment.rl.prompts import (
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


def init_wandb(train_config: TrainConfig, wandb_config: WandbConfig):
  if not wandb_config.enabled:
    return None
  if wandb_config.api_key is not None:
    os.environ.setdefault("WANDB_API_KEY", wandb_config.api_key)
  try:
    import wandb
  except ImportError as error:
    raise RuntimeError("wandb is not installed. Install the plots or gpu extra, or omit --wandb-project.") from error

  return wandb.init(
    project=wandb_config.project,
    entity=wandb_config.entity,
    name=wandb_config.run_name,
    mode=wandb_config.mode,
    config=train_config.to_dict(),
  )


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
  if not examples:
    raise ValueError(f"No examples loaded from {path}")
  logger.info("Loaded %d examples", len(examples))
  return examples


def main() -> None:
  parsed_config = parse_train_config()
  config = parsed_config.train
  wandb_config = parsed_config.wandb
  logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
  )
  torch.manual_seed(0)
  device = torch.device(config.device)
  logger.info("Starting local GRPO smoke test")
  logger.info("Config: %s", config.to_json())
  wandb_run = init_wandb(config, wandb_config)

  examples = load_gsm8k_examples(config.data_path, config.n_train_examples)
  rollout_prompt_count = config.num_rollout_prompts()
  prompt = get_prompt(config.prompt)
  logger.info(
    "Rollout setup: rollout_batch_size=%d rollout_prompts=%d grad_accum=%d",
    config.rollout_batch_size,
    rollout_prompt_count,
    config.gradient_accumulation_steps,
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
    "lr": config.lr,
    "weight_decay": config.weight_decay,
    **config.optimizer_params,
  }
  if config.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
  elif config.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
  else:
    raise ValueError(f"Unknown optimizer: {config.optimizer}")
  num_params = sum(param.numel() for param in model.parameters())
  logger.info("Model ready on %s with %d parameters and context length %d", device, num_params, model.config.n_positions)
  logger.info("Optimizer ready: %s params=%s", optimizer.__class__.__name__, optimizer_kwargs)
  if wandb_run is not None:
    wandb_run.log(
      {
        "model/num_params": num_params,
        "model/context_length": model.config.n_positions,
        "rollout/rollout_prompt_count": rollout_prompt_count,
      },
      step=0,
    )
  logger.info("Running GRPO train steps")

  for step in progress(range(config.num_rollout_steps), desc="training", unit="step"):
    logger.debug("Starting step %d", step)
    start = (step * rollout_prompt_count) % len(examples)
    batch_examples = [examples[(start + i) % len(examples)] for i in range(rollout_prompt_count)]

    if config.inference is None:
      prompts, outputs, ground_truths = make_prompt_rollouts(prompt, batch_examples, config.group_size)
    else:
      logger.info("Generating rollouts from inference endpoint: %s", config.inference)
      prompts, outputs, ground_truths = make_vllm_rollouts(
        config.inference,
        config.vllm_model,
        prompt,
        batch_examples,
        config.group_size,
        config.sampling_temperature,
        config.sampling_max_tokens,
        config.seed + step,
        config.inference_batch_size,
      )
    logger.info(
      "Built rollout batch: step=%d prompt=%s prompts=%d outputs=%d ground_truths=%d reward_fn=%s",
      step,
      config.prompt,
      len(prompts),
      len(outputs),
      len(ground_truths),
      prompt.reward_fn.__name__,
    )

    result = grpo_train_step(
      model=model,
      tokenizer=tokenizer,
      optimizer=optimizer,
      reward_fn=prompt.reward_fn,
      prompt_strs=prompts,
      output_strs=outputs,
      ground_truths=ground_truths,
      group_size=config.group_size,
      gradient_accumulation_steps=config.gradient_accumulation_steps,
      max_grad_norm=config.max_grad_norm,
    )
    metrics = {
      "train/loss": result.loss.item(),
      "reward/mean": result.rewards.float().mean().item(),
      "reward/min": result.rewards.float().min().item(),
      "reward/max": result.rewards.float().max().item(),
      "advantage/mean": result.advantages.float().mean().item(),
      "advantage/std": result.advantages.float().std(unbiased=False).item(),
      "rollout/batch_size": len(outputs),
      "rollout/prompt_count": len(batch_examples),
      "rollout/log_prob_seq_len": result.log_probs.shape[1],
    }
    if wandb_run is not None:
      wandb_run.log(metrics, step=step)

    message = (
      "step={step} loss={loss:.6f} reward_mean={reward_mean:.3f} "
      "reward_min={reward_min:.3f} reward_max={reward_max:.3f} "
      "advantage_mean={advantage_mean:.3f} advantage_std={advantage_std:.3f} "
      "log_prob_shape={log_prob_shape}".format(
        step=step,
        loss=metrics["train/loss"],
        reward_mean=metrics["reward/mean"],
        reward_min=metrics["reward/min"],
        reward_max=metrics["reward/max"],
        advantage_mean=metrics["advantage/mean"],
        advantage_std=metrics["advantage/std"],
        log_prob_shape=tuple(result.log_probs.shape),
      )
    )
    logger.info(message)

  if wandb_run is not None:
    wandb_run.finish()
  logger.info("Done")


if __name__ == "__main__":
  main()
