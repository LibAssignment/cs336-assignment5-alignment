from __future__ import annotations

import atexit
import json
import logging
import os
import stat
import sys
from pathlib import Path
from typing import Iterable, TypeVar

import torch

from .models import build_model_and_tokenizer, model_context_length
from .grpo import grpo_train_step
from .config import JobConfig, TrainConfig, WandbConfig
from .jobs import Job, checkpoint_dir, clear_pid, latest_checkpoint, prepare_job, write_json, write_latest, write_pid, write_status
from .prompts import (
  extract_gsm8k_answer,
  get_prompt,
  make_smoke_rollouts,
  make_vllm_rollouts,
)


T = TypeVar("T")
logger = logging.getLogger(__name__)


def is_regular_file(stream) -> bool:
  try:
    return stat.S_ISREG(os.fstat(stream.fileno()).st_mode)
  except (AttributeError, OSError):
    return False


def is_devnull(stream) -> bool:
  try:
    stream_stat = os.fstat(stream.fileno())
    devnull_stat = os.stat(os.devnull)
  except (AttributeError, OSError):
    return False
  return stat.S_ISCHR(stream_stat.st_mode) and stream_stat.st_rdev == devnull_stat.st_rdev


def suppress_console_output() -> bool:
  return is_regular_file(sys.stderr) and is_devnull(sys.stdin)



def progress(iterable: Iterable[T], **kwargs) -> Iterable[T]:
  if suppress_console_output():
    return iterable
  try:
    from tqdm.auto import tqdm
  except ImportError:
    return iterable
  return tqdm(iterable, **kwargs)


class TrainState:
  def __init__(
    self,
    config: TrainConfig,
    wandb_config: WandbConfig | None = None,
    job_config: JobConfig | None = None,
  ) -> None:
    self.config = config
    self.wandb_config = wandb_config
    self.job_config = job_config
    self.job = prepare_job(job_config, config, wandb_config) if job_config is not None else None
    self.wandb_run = None
    self.completed = False

  def init(self) -> None:
    self.configure_logging()
    if self.job is not None:
      write_pid(self.job)
      write_status(self.job, "running", pid=os.getpid())
      logger.info("Job %d output: %s", self.job.run_id, self.job.path)
      atexit.register(self.mark_failed_on_exit)

    logger.info("Starting local GRPO smoke test")
    logger.info("Config: %s", self.config.to_json())
    self.wandb_run = self.init_wandb()
    self.save_wandb_run_name()

  def configure_logging(self) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, self.config.log_level))
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not suppress_console_output():
      stream_handler = logging.StreamHandler()
      stream_handler.setLevel(getattr(logging, self.config.log_level))
      stream_handler.setFormatter(formatter)
      root.addHandler(stream_handler)
    if self.job is not None:
      file_handler = logging.FileHandler(self.job.log_path)
      file_handler.setLevel(getattr(logging, self.config.log_level))
      file_handler.setFormatter(formatter)
      root.addHandler(file_handler)

  def init_wandb(self):
    if self.wandb_config is None or not self.wandb_config.enabled:
      return None
    if self.wandb_config.api_key is not None:
      os.environ.setdefault("WANDB_API_KEY", self.wandb_config.api_key)
    try:
      import wandb
    except ImportError as error:
      raise RuntimeError("wandb is not installed. Install the plots or gpu extra, or omit --wandb-project.") from error

    return wandb.init(
      project=self.wandb_config.project,
      entity=self.wandb_config.entity,
      name=self.wandb_config.run_name,
      mode=self.wandb_config.mode,  # type: ignore
      config=self.config.to_dict(),
    )

  def save_wandb_run_name(self) -> None:
    if self.job is None or self.wandb_config is None or self.wandb_run is None:
      return
    run_name = getattr(self.wandb_run, "name", None)
    if not isinstance(run_name, str) or not run_name:
      return
    self.wandb_config.run_name = run_name
    write_json(self.job.path / "wandb_config.json", self.wandb_config.to_dict())

  def mark_failed_on_exit(self) -> None:
    if self.job is not None and not self.completed:
      clear_pid(self.job)
      write_status(self.job, "failed", exit_code=1)

  def step_metrics(self, values: dict[str, float | int], step: int) -> None:
    if self.wandb_run is not None:
      self.wandb_run.log(values, step=step)
    message = f"step={step}"
    for key, value in values.items():
      key = key.replace("/", "_")
      message += f" {key}={value:.6f}" if isinstance(value, float) else f" {key}={value}"
    logger.info(message)

  def model_metrics(self, num_params: int, rollout_prompt_count: int, context_length: int | None) -> None:
    if self.wandb_run is None:
      return
    values = {
      "model/num_params": num_params,
      "rollout/rollout_prompt_count": rollout_prompt_count,
    }
    if context_length is not None:
      values["model/context_length"] = context_length
    self.wandb_run.log(values, step=0)

  def save_checkpoint(self, step: int, model, optimizer) -> Path | None:
    if self.job is None or self.job_config is None or self.job_config.checkpoint_every <= 0:
      return None
    if step % self.job_config.checkpoint_every != 0 and step != self.config.num_rollout_steps:
      return None
    path = checkpoint_dir(self.job, step)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    (path / "trainer_state.json").write_text(json.dumps({"next_step": step}, indent=2, sort_keys=True) + "\n")
    write_latest(self.job, path)
    write_status(self.job, "running", checkpoint=str(path.relative_to(self.job.path)), next_step=step)
    logger.info("Saved checkpoint: %s", path)
    return path

  def load_resume_checkpoint(self, model, optimizer, device: torch.device) -> int:
    if self.job is None or self.job_config is None or not self.job_config.resume:
      return 0
    checkpoint = latest_checkpoint(self.job)
    if checkpoint is None:
      raise FileNotFoundError(f"No latest checkpoint found for job {self.job.run_id} at {self.job.path}")
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device))
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device))
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
    next_step = int(trainer_state["next_step"])
    logger.info("Loaded checkpoint: %s next_step=%d", checkpoint, next_step)
    return next_step

  def progress(self, iterable: Iterable[T], **kwargs) -> Iterable[T]:
    return progress(iterable, **kwargs)

  def finish(self) -> None:
    if self.wandb_run is not None:
      self.wandb_run.finish()
    if self.job is not None:
      self.completed = True
      clear_pid(self.job)
      write_status(self.job, "succeeded", exit_code=0)
    logger.info("Done")


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


def train(config: TrainConfig, wandb_config: WandbConfig | None = None, job_config: JobConfig | None = None) -> None:
  state = TrainState(config, wandb_config, job_config)
  state.init()
  torch.manual_seed(0)
  device = torch.device(config.device)

  examples = load_gsm8k_examples(config.data_path, config.n_train_examples)
  rollout_prompt_count = config.num_rollout_prompts()
  prompt = get_prompt(config.prompt)
  logger.info(
    "Rollout setup: rollout_batch_size=%d rollout_prompts=%d grad_accum=%d",
    config.rollout_batch_size,
    rollout_prompt_count,
    config.gradient_accumulation_steps,
  )
  logger.info("Creating model=%s on %s", config.model, device)
  model, tokenizer = build_model_and_tokenizer(config, device)
  logger.info(
    "Tokenizer ready: vocab_size=%d eos_token_id=%d pad_token_id=%d ",
    len(tokenizer),
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
  )
  optimizer_kwargs = {
    "lr": config.lr,
    "weight_decay": config.weight_decay,
    **config.optimizer_params,
  }
  if config.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)  # type: ignore
  elif config.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)  # type: ignore
  else:
    raise ValueError(f"Unknown optimizer: {config.optimizer}")
  num_params = sum(param.numel() for param in model.parameters())
  context_length = model_context_length(model)
  logger.info("Model ready on %s with %d parameters and context length %s", device, num_params, context_length or "unknown")
  logger.info("Optimizer ready: %s params=%s", optimizer.__class__.__name__, optimizer_kwargs)
  start_step = state.load_resume_checkpoint(model, optimizer, device)
  state.model_metrics(num_params, rollout_prompt_count, context_length)
  logger.info("Running GRPO train steps")

  if config.inference == "vllm":
    logger.info("Initializing weight sync with vLLM inference engine at %s", config.inference)
    from ..vllm_utils import VLLMServer, init_weight_sync
    model_id = config.model_path() or config.vllm_model
    vllm = VLLMServer(str(model_id), config.vllm_model)

    init_weight_sync(config.inference, config.device)
  else:
    vllm = None

  for step in state.progress(range(start_step, config.num_rollout_steps), desc="training", unit="step"):
    logger.debug("Starting step %d", step)
    start = (step * rollout_prompt_count) % len(examples)
    batch_examples = [examples[(start + i) % len(examples)] for i in range(rollout_prompt_count)]

    if config.inference is None or config.inference == "smoke":
      prompts, outputs, ground_truths = make_smoke_rollouts(prompt, batch_examples, config.group_size)
    else:
      logger.info("Generating rollouts from inference endpoint: %s", config.inference)
      prompts, outputs, ground_truths = make_vllm_rollouts(
        vllm or config.inference,
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
    state.step_metrics(metrics, step=step)
    state.save_checkpoint(step + 1, model, optimizer)

  state.finish()
