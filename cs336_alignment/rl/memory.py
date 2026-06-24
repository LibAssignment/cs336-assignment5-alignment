from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from cs336_alignment.rl.utils import tokenize_prompt_and_output


DEFAULT_OLMO2_1B_MODEL_GIB = 3.5
DEFAULT_OLMO2_1B_ADAM_SCALE = 4.0
DEFAULT_OLMO2_1B_LAYER_ACT_GIB = 1.19
DEFAULT_OLMO2_1B_LOGITS_GIB = 1.53
DEFAULT_OLMO2_1B_LM_HEAD_SCALE = 4.0
DEFAULT_OLMO2_1B_RESULT_SCALE = 3.5
DEFAULT_OLMO2_1B_MEMORY_TOKENS = 8 * 1024
DEFAULT_MEMORY_UTILIZATION = 0.9


@dataclass
class MemoryEstimate:
  model_gib: float | None = None
  adam_scale: float = DEFAULT_OLMO2_1B_ADAM_SCALE
  layer_act_gib: float | None = None
  logits_gib: float | None = None
  lm_head_scale: float = DEFAULT_OLMO2_1B_LM_HEAD_SCALE
  result_scale: float = DEFAULT_OLMO2_1B_RESULT_SCALE
  tokens: int | None = None
  utilization: float = DEFAULT_MEMORY_UTILIZATION

  def all_none(self) -> bool:
    return (
      self.model_gib is None
      and self.layer_act_gib is None
      and self.logits_gib is None
    )

  def complete(self) -> bool:
    return (
      self.model_gib is not None
      and self.layer_act_gib is not None
      and self.logits_gib is not None
    )

  def valid(self) -> bool:
    return (
      self.model_gib is not None
      and self.layer_act_gib is not None
      and self.logits_gib is not None
      and self.model_gib > 0
      and self.adam_scale > 0
      and self.layer_act_gib > 0
      and self.logits_gib > 0
      and self.lm_head_scale >= 0
      and self.result_scale >= 0
      and (self.tokens is None or self.tokens > 0)
      and self.utilization > 0
      and self.utilization <= 1
    )


@dataclass
class MemoryEstimateResult:
  seq_len: int
  macrobatch_size: int
  macro_adam_gib: float = 0.0
  macro_act_gib: float = 0.0
  macro_var_gib: float = 0.0
  macro_total_gib: float = 0.0
  trimmed_macrobatch: bool = False


DEFAULT_OLMO2_1B_MEMORY_ESTIMATE = MemoryEstimate(
  model_gib=DEFAULT_OLMO2_1B_MODEL_GIB,
  adam_scale=DEFAULT_OLMO2_1B_ADAM_SCALE,
  layer_act_gib=DEFAULT_OLMO2_1B_LAYER_ACT_GIB,
  logits_gib=DEFAULT_OLMO2_1B_LOGITS_GIB,
  lm_head_scale=DEFAULT_OLMO2_1B_LM_HEAD_SCALE,
  result_scale=DEFAULT_OLMO2_1B_RESULT_SCALE,
  tokens=DEFAULT_OLMO2_1B_MEMORY_TOKENS,
  utilization=DEFAULT_MEMORY_UTILIZATION,
)


def default_memory_estimate_for_model(model: str) -> MemoryEstimate | None:
  if model != "olmo2-1B":
    return None
  return MemoryEstimate(
    model_gib=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.model_gib,
    adam_scale=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.adam_scale,
    layer_act_gib=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.layer_act_gib,
    logits_gib=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.logits_gib,
    lm_head_scale=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.lm_head_scale,
    result_scale=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.result_scale,
    tokens=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.tokens,
    utilization=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.utilization,
  )


def model_layer_count(model: Any) -> int:
  config = getattr(model, "config", None)
  return int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0) or 0)


def token_scale(memory_estimate: MemoryEstimate, tokens: int) -> float:
  reference_tokens = memory_estimate.tokens or tokens
  return tokens / reference_tokens


def estimate_adam_memory_gib(memory_estimate: MemoryEstimate | None) -> float | None:
  if memory_estimate is None or not memory_estimate.valid():
    return None
  return memory_estimate.model_gib * memory_estimate.adam_scale


def estimate_activation_memory_gib(
  model: Any,
  memory_estimate: MemoryEstimate | None,
  *,
  tokens: int,
) -> float | None:
  if memory_estimate is None or not memory_estimate.valid():
    return None
  num_layers = model_layer_count(model)
  if num_layers <= 0:
    return None
  return num_layers * memory_estimate.layer_act_gib * token_scale(memory_estimate, tokens)


def estimate_variable_memory_gib(memory_estimate: MemoryEstimate | None, *, tokens: int) -> float | None:
  if memory_estimate is None or not memory_estimate.valid():
    return None
  return (
    memory_estimate.logits_gib
    * (memory_estimate.lm_head_scale + memory_estimate.result_scale)
    * token_scale(memory_estimate, tokens)
  )


def estimate_rollout_memory(
  model: Any,
  tokenizer: Any,
  prompt_strs: list[str],
  output_strs: list[str],
  memory_estimate: MemoryEstimate,
) -> MemoryEstimateResult:
  tokenized = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
  seq_len = tokenized.input_ids.shape[1]
  requested_macrobatch_size = len(prompt_strs)
  macrobatch_size = trim_macrobatch_size_to_memory(
    model,
    requested_macrobatch_size,
    seq_len=seq_len,
    memory_estimate=memory_estimate,
  )
  macro_tokens = macrobatch_size * seq_len
  macro_adam_gib = 0.0
  macro_act_gib = 0.0
  macro_var_gib = 0.0
  macro_total_gib = 0.0
  if memory_estimate.valid():
    macro_adam_gib = estimate_adam_memory_gib(memory_estimate) or 0.0
    macro_act_gib = estimate_activation_memory_gib(model, memory_estimate, tokens=macro_tokens) or 0.0
    macro_var_gib = estimate_variable_memory_gib(memory_estimate, tokens=macro_tokens) or 0.0
    macro_total_gib = macro_adam_gib + macro_act_gib + macro_var_gib

  return MemoryEstimateResult(
    seq_len=seq_len,
    macrobatch_size=macrobatch_size,
    macro_adam_gib=macro_adam_gib,
    macro_act_gib=macro_act_gib,
    macro_var_gib=macro_var_gib,
    macro_total_gib=macro_total_gib,
    trimmed_macrobatch=macrobatch_size < requested_macrobatch_size,
  )


def trim_macrobatch_size_to_memory(
  model: Any,
  macrobatch_size: int,
  *,
  seq_len: int,
  memory_estimate: MemoryEstimate | None = None,
) -> int:
  if (
    memory_estimate is None
    or not memory_estimate.valid()
    or seq_len <= 0
    or macrobatch_size <= 1
    or not torch.cuda.is_available()
  ):
    return macrobatch_size

  num_layers = model_layer_count(model)
  if num_layers <= 0:
    return macrobatch_size

  device = model.device
  if device.type != "cuda":
    return macrobatch_size

  _, total_bytes = torch.cuda.mem_get_info(device)
  total_gib = total_bytes / (1024**3)
  adam_gib = estimate_adam_memory_gib(memory_estimate)
  act_gib = estimate_activation_memory_gib(model, memory_estimate, tokens=macrobatch_size * seq_len)
  var_gib = estimate_variable_memory_gib(memory_estimate, tokens=macrobatch_size * seq_len)
  if adam_gib is None or act_gib is None or var_gib is None:
    return macrobatch_size
  variable_budget_gib = total_gib * memory_estimate.utilization - adam_gib
  requested_variable_gib = act_gib + var_gib
  if variable_budget_gib <= 0 or requested_variable_gib <= 0:
    return 1

  max_macrobatch_size = max(1, math.floor(macrobatch_size * variable_budget_gib / requested_variable_gib))
  return min(macrobatch_size, max_macrobatch_size)
