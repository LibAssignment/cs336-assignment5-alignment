from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cs336_alignment.rl.prompts import PROMPT_KINDS


def parse_optimizer_params(raw_params: str) -> dict[str, Any]:
  try:
    params = json.loads(raw_params)
  except json.JSONDecodeError as error:
    raise argparse.ArgumentTypeError(f"Invalid optimizer params JSON: {error}") from error
  if not isinstance(params, dict):
    raise argparse.ArgumentTypeError("--optimizer-params must decode to a JSON object")
  if "betas" in params:
    params["betas"] = tuple(params["betas"])
  return params


def inference_base_url(inference: str | None) -> str | None:
  if inference is None or inference == "smoke":
    return None
  if inference == "vllm":
    return os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
  if inference.startswith("http://") or inference.startswith("https://"):
    return inference
  raise argparse.ArgumentTypeError("--inference must be 'smoke', 'vllm', or an http(s) URL")


@dataclass
class TrainConfig:
  data_path: Path = Path("data/gsm8k/train.jsonl")
  prompt: str = "question_only"
  inference: str | None = None
  vllm_model: str = ""
  inference_batch_size: int = 5
  sampling_temperature: float = 1.0
  sampling_max_tokens: int = 512
  seed: int = 42
  n_train_examples: int = 6400
  n_val_examples: int = 1024
  rollout_batch_size: int = 256
  group_size: int = 2
  num_rollout_steps: int = 1
  device: str = "cpu"
  optimizer: str = "adamw"
  lr: float = 1e-3
  weight_decay: float = 0.0
  optimizer_params: dict[str, Any] = field(default_factory=dict)
  gradient_accumulation_steps: int = 32
  max_grad_norm: float = 1.0
  log_level: str = "INFO"

  def to_dict(self) -> dict[str, Any]:
    result = asdict(self)
    result["data_path"] = str(self.data_path)
    return result

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
    config_data = dict(data)
    if "temperature" in config_data:
      config_data["sampling_temperature"] = config_data.pop("temperature")
    if "max_tokens" in config_data:
      config_data["sampling_max_tokens"] = config_data.pop("max_tokens")
    if "steps" in config_data:
      config_data["num_rollout_steps"] = config_data.pop("steps")
    if "data_path" in config_data:
      config_data["data_path"] = Path(config_data["data_path"])
    if "inference" in config_data:
      config_data["inference"] = inference_base_url(config_data["inference"])
    if config_data.get("optimizer_params") is None:
      config_data["optimizer_params"] = {}
    if "betas" in config_data.get("optimizer_params", {}):
      config_data["optimizer_params"]["betas"] = tuple(config_data["optimizer_params"]["betas"])
    return cls(**config_data)

  def num_rollout_prompts(self) -> int:
    if self.rollout_batch_size % self.group_size != 0:
      raise ValueError("rollout_batch_size must be divisible by group_size")
    return self.rollout_batch_size // self.group_size

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), indent=2, sort_keys=True)

  @classmethod
  def from_json(cls, raw: str) -> TrainConfig:
    data = json.loads(raw)
    if not isinstance(data, dict):
      raise ValueError("TrainConfig JSON must decode to an object")
    return cls.from_dict(data)

  def save_json(self, path: Path) -> None:
    path.write_text(self.to_json() + "\n")

  @classmethod
  def load_json(cls, path: Path) -> TrainConfig:
    return cls.from_json(path.read_text())


def add_train_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
  parser.add_argument("--config", type=Path, default=None)
  parser.add_argument("--save-config", type=Path, default=None)
  parser.add_argument("--data-path", type=Path, default=argparse.SUPPRESS)
  parser.add_argument("--prompt", choices=PROMPT_KINDS, default=argparse.SUPPRESS)
  parser.add_argument("--inference", type=inference_base_url, default=argparse.SUPPRESS)
  parser.add_argument("--vllm-model", default=argparse.SUPPRESS)
  parser.add_argument("--inference-batch-size", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--sampling-temperature", type=float, default=argparse.SUPPRESS)
  parser.add_argument("--temperature", type=float, dest="sampling_temperature", default=argparse.SUPPRESS)
  parser.add_argument("--sampling-max-tokens", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--max-tokens", type=int, dest="sampling_max_tokens", default=argparse.SUPPRESS)
  parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--n-train-examples", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--n-val-examples", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--rollout-batch-size", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--group-size", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--num-rollout-steps", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--steps", type=int, dest="num_rollout_steps", default=argparse.SUPPRESS)
  parser.add_argument("--device", default=argparse.SUPPRESS)
  parser.add_argument("--optimizer", choices=["adamw", "sgd"], default=argparse.SUPPRESS)
  parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
  parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
  parser.add_argument(
    "--optimizer-params",
    type=parse_optimizer_params,
    default=argparse.SUPPRESS,
    help='Extra optimizer kwargs as JSON, e.g. \'{"betas":[0.9,0.95]}\' for AdamW.',
  )
  parser.add_argument("--gradient-accumulation-steps", type=int, default=argparse.SUPPRESS)
  parser.add_argument("--max-grad-norm", type=float, default=argparse.SUPPRESS)
  parser.add_argument("--log-level", default=argparse.SUPPRESS, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
  return parser


def parse_train_config(argv: list[str] | None = None) -> TrainConfig:
  parser = argparse.ArgumentParser(description="Local smoke test for grpo_train_step on GSM8K.")
  add_train_config_args(parser)
  args = parser.parse_args(argv)

  config = TrainConfig.load_json(args.config) if args.config is not None else TrainConfig()
  for key, value in vars(args).items():
    if key in {"config", "save_config"}:
      continue
    setattr(config, key, value)

  if args.save_config is not None:
    config.save_json(args.save_config)

  return config
