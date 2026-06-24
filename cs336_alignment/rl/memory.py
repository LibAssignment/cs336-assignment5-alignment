from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


DEFAULT_OLMO2_1B_MEMORY_PER_LAYER_GIB = 1.19
DEFAULT_OLMO2_1B_MEMORY_HEAD_GIB = 6.12
DEFAULT_OLMO2_1B_MEMORY_PARAMS_GIB = 14.0
DEFAULT_MEMORY_UTILIZATION = 0.9


@dataclass
class MemoryEstimate:
  per_layer_gib: float | None = None
  head_gib: float | None = None
  params_gib: float | None = None
  utilization: float = DEFAULT_MEMORY_UTILIZATION

  def all_none(self) -> bool:
    return self.per_layer_gib is None and self.head_gib is None and self.params_gib is None

  def complete(self) -> bool:
    return self.per_layer_gib is not None and self.head_gib is not None and self.params_gib is not None

  def valid(self) -> bool:
    return (
      self.per_layer_gib is not None
      and self.head_gib is not None
      and self.params_gib is not None
      and self.per_layer_gib > 0
      and self.head_gib > 0
      and self.params_gib > 0
      and self.utilization > 0
      and self.utilization <= 1
    )


DEFAULT_OLMO2_1B_MEMORY_ESTIMATE = MemoryEstimate(
  per_layer_gib=DEFAULT_OLMO2_1B_MEMORY_PER_LAYER_GIB,
  head_gib=DEFAULT_OLMO2_1B_MEMORY_HEAD_GIB,
  params_gib=DEFAULT_OLMO2_1B_MEMORY_PARAMS_GIB,
  utilization=DEFAULT_MEMORY_UTILIZATION,
)


def default_memory_estimate_for_model(model: str) -> MemoryEstimate | None:
  if model != "olmo2-1B":
    return None
  return MemoryEstimate(
    per_layer_gib=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.per_layer_gib,
    head_gib=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.head_gib,
    params_gib=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.params_gib,
    utilization=DEFAULT_OLMO2_1B_MEMORY_ESTIMATE.utilization,
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

  config = getattr(model, "config", None)
  num_layers = int(getattr(config, "num_hidden_layers", 0) or getattr(config, "n_layer", 0) or 0)
  if num_layers <= 0:
    return macrobatch_size

  device = model.device
  if device.type != "cuda":
    return macrobatch_size

  _, total_bytes = torch.cuda.mem_get_info(device)
  total_gib = total_bytes / (1024**3)
  variable_budget_gib = total_gib * memory_estimate.utilization - memory_estimate.params_gib
  requested_variable_gib = num_layers * memory_estimate.per_layer_gib + memory_estimate.head_gib
  if variable_budget_gib <= 0 or requested_variable_gib <= 0:
    return 1

  max_macrobatch_size = max(1, math.floor(macrobatch_size * variable_budget_gib / requested_variable_gib))
  return min(macrobatch_size, max_macrobatch_size)
