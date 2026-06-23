from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM

from cs336_alignment.rl.config import DEFAULT_OLMO2_1B_PATH
from cs336_alignment.rl.models import tiny_byte_tokenizer, tiny_train_model


BYTES_PER_GIB = 1024**3


@dataclass
class TrialResult:
  optimizer: str
  batch_size: int
  seq_len: int
  status: str
  seconds: float | None
  cuda_baseline_allocated_gib: float | None
  cuda_peak_allocated_gib: float | None
  cuda_peak_extra_gib: float | None
  cuda_peak_reserved_gib: float | None
  cuda_reserved_after_gib: float | None
  cuda_free_after_gib: float | None
  host_rss_after_gib: float
  error: str | None = None


def gib(num_bytes: int | float | None) -> float | None:
  if num_bytes is None:
    return None
  return num_bytes / BYTES_PER_GIB


def parse_int_list(raw: str) -> list[int]:
  values = []
  for item in raw.split(","):
    item = item.strip()
    if item:
      values.append(int(item))
  if not values:
    raise argparse.ArgumentTypeError("expected at least one integer")
  return values


def proc_rss_bytes() -> int:
  status_path = Path("/proc/self/status")
  if status_path.exists():
    for line in status_path.read_text().splitlines():
      if line.startswith("VmRSS:"):
        return int(line.split()[1]) * 1024
  try:
    import resource

    # ru_maxrss is KiB on Linux and bytes on macOS. This script is intended for
    # the Linux CUDA host, but keep the fallback sane enough elsewhere.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if sys.platform.startswith("linux") else value)
  except Exception:
    return 0


def host_ram_bytes() -> int | None:
  meminfo_path = Path("/proc/meminfo")
  if meminfo_path.exists():
    for line in meminfo_path.read_text().splitlines():
      if line.startswith("MemTotal:"):
        return int(line.split()[1]) * 1024
  if hasattr(os, "sysconf"):
    try:
      return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
      return None
  return None


def cuda_device_report(device: torch.device) -> dict[str, object]:
  index = device.index if device.index is not None else torch.cuda.current_device()
  props = torch.cuda.get_device_properties(index)
  free_bytes, total_bytes = torch.cuda.mem_get_info(index)
  return {
    "index": index,
    "name": props.name,
    "total_gib": gib(total_bytes),
    "free_gib": gib(free_bytes),
    "capability": f"{props.major}.{props.minor}",
  }


def optimizer_for(name: str, model: torch.nn.Module, lr: float) -> torch.optim.Optimizer | None:
  if name == "none":
    return None
  if name == "adamw":
    return torch.optim.AdamW(model.parameters(), lr=lr)
  if name == "sgd":
    return torch.optim.SGD(model.parameters(), lr=lr)
  raise ValueError(f"unknown optimizer: {name}")


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, int]:
  if args.model == "tiny":
    tokenizer = tiny_byte_tokenizer()
    model = tiny_train_model(tokenizer, n_positions=max(args.seq_lens), device=device)
    return model, len(tokenizer)

  model_path = (args.model_path or DEFAULT_OLMO2_1B_PATH).expanduser()
  dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
  model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    device_map=str(device),
    torch_dtype=dtype,
    attn_implementation="flash_attention_2" if args.flash_attention else "eager",
  )
  model.train()
  vocab_size = int(getattr(model.config, "vocab_size"))
  return model, vocab_size


def reset_cuda_peak(device: torch.device) -> None:
  torch.cuda.synchronize(device)
  torch.cuda.reset_peak_memory_stats(device)


def random_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  # Avoid special-token assumptions; random token ids are enough to exercise the
  # same embedding, transformer, logits, loss, and backward memory paths.
  input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
  labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
  return input_ids, labels


def run_trial(
  model: torch.nn.Module,
  vocab_size: int,
  optimizer_name: str,
  batch_size: int,
  seq_len: int,
  device: torch.device,
  args: argparse.Namespace,
) -> TrialResult:
  optimizer = optimizer_for(optimizer_name, model, args.lr)
  input_ids = None
  labels = None
  attention_mask = None
  outputs = None
  logits = None
  loss = None
  probs = None
  entropy = None
  gc.collect()
  torch.cuda.empty_cache()
  baseline_allocated = torch.cuda.memory_allocated(device)
  reset_cuda_peak(device)
  start = time.perf_counter()
  try:
    input_ids, labels = random_batch(batch_size, seq_len, vocab_size, device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    loss = torch.nn.functional.cross_entropy(
      logits.reshape(-1, logits.size(-1)),
      labels.reshape(-1),
      reduction="mean",
    )
    if args.include_entropy:
      probs = torch.nn.functional.softmax(logits, dim=-1)
      entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
      loss = loss + args.entropy_weight * entropy
    loss.backward()
    if optimizer is not None:
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
    else:
      model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    free_after, _ = torch.cuda.mem_get_info(device)
    return TrialResult(
      optimizer=optimizer_name,
      batch_size=batch_size,
      seq_len=seq_len,
      status="ok",
      seconds=seconds,
      cuda_baseline_allocated_gib=gib(baseline_allocated),
      cuda_peak_allocated_gib=gib(torch.cuda.max_memory_allocated(device)),
      cuda_peak_extra_gib=gib(torch.cuda.max_memory_allocated(device) - baseline_allocated),
      cuda_peak_reserved_gib=gib(torch.cuda.max_memory_reserved(device)),
      cuda_reserved_after_gib=gib(torch.cuda.memory_reserved(device)),
      cuda_free_after_gib=gib(free_after),
      host_rss_after_gib=gib(proc_rss_bytes()) or 0.0,
    )
  except torch.OutOfMemoryError as error:
    model.zero_grad(set_to_none=True)
    if optimizer is not None:
      optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    return TrialResult(
      optimizer=optimizer_name,
      batch_size=batch_size,
      seq_len=seq_len,
      status="oom",
      seconds=None,
      cuda_baseline_allocated_gib=gib(baseline_allocated),
      cuda_peak_allocated_gib=gib(torch.cuda.max_memory_allocated(device)),
      cuda_peak_extra_gib=gib(torch.cuda.max_memory_allocated(device) - baseline_allocated),
      cuda_peak_reserved_gib=gib(torch.cuda.max_memory_reserved(device)),
      cuda_reserved_after_gib=gib(torch.cuda.memory_reserved(device)),
      cuda_free_after_gib=gib(free_after),
      host_rss_after_gib=gib(proc_rss_bytes()) or 0.0,
      error=str(error).splitlines()[0],
    )
  except RuntimeError as error:
    model.zero_grad(set_to_none=True)
    if optimizer is not None:
      optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    return TrialResult(
      optimizer=optimizer_name,
      batch_size=batch_size,
      seq_len=seq_len,
      status="error",
      seconds=None,
      cuda_baseline_allocated_gib=gib(baseline_allocated),
      cuda_peak_allocated_gib=gib(torch.cuda.max_memory_allocated(device)),
      cuda_peak_extra_gib=gib(torch.cuda.max_memory_allocated(device) - baseline_allocated),
      cuda_peak_reserved_gib=gib(torch.cuda.max_memory_reserved(device)),
      cuda_reserved_after_gib=gib(torch.cuda.memory_reserved(device)),
      cuda_free_after_gib=gib(free_after),
      host_rss_after_gib=gib(proc_rss_bytes()) or 0.0,
      error=str(error).splitlines()[0],
    )
  finally:
    del input_ids, labels, attention_mask, outputs, logits, loss, probs, entropy, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def print_table(results: list[TrialResult]) -> None:
  header = (
    f"{'opt':<6} {'B':>4} {'S':>6} {'status':<6} {'sec':>7} "
    f"{'base_alloc':>11} {'peak_alloc':>11} {'peak_extra':>11} "
    f"{'peak_reserved':>13} {'reserved_after':>14} {'free_after':>11} {'host_rss':>10}"
  )
  print(header)
  print("-" * len(header))
  for row in results:
    seconds = "" if row.seconds is None else f"{row.seconds:.2f}"
    base_alloc = "" if row.cuda_baseline_allocated_gib is None else f"{row.cuda_baseline_allocated_gib:.2f}"
    peak_alloc = "" if row.cuda_peak_allocated_gib is None else f"{row.cuda_peak_allocated_gib:.2f}"
    peak_extra = "" if row.cuda_peak_extra_gib is None else f"{row.cuda_peak_extra_gib:.2f}"
    peak_reserved = "" if row.cuda_peak_reserved_gib is None else f"{row.cuda_peak_reserved_gib:.2f}"
    reserved_after = "" if row.cuda_reserved_after_gib is None else f"{row.cuda_reserved_after_gib:.2f}"
    free_after = "" if row.cuda_free_after_gib is None else f"{row.cuda_free_after_gib:.2f}"
    print(
      f"{row.optimizer:<6} {row.batch_size:>4} {row.seq_len:>6} {row.status:<6} {seconds:>7} "
      f"{base_alloc:>11} {peak_alloc:>11} {peak_extra:>11} "
      f"{peak_reserved:>13} {reserved_after:>14} {free_after:>11} {row.host_rss_after_gib:>10.2f}"
    )
    if row.error:
      print(f"  error: {row.error}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Measure real forward/backward memory over batch-size and sequence-length grids.")
  parser.add_argument("--model", choices=["tiny", "olmo2-1B"], default="olmo2-1B")
  parser.add_argument("--model-path", type=Path, default=None)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("1,2,4,8"))
  parser.add_argument("--seq-lens", type=parse_int_list, default=parse_int_list("256,512,768,1024"))
  parser.add_argument("--optimizers", type=lambda raw: [item.strip() for item in raw.split(",") if item.strip()], default=["adamw"])
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
  parser.add_argument("--flash-attention", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--include-entropy", action="store_true", help="Also compute the full-vocab entropy term used by the current GRPO helper.")
  parser.add_argument("--entropy-weight", type=float, default=0.0)
  parser.add_argument("--json", type=Path, default=None, help="Optional path to write machine-readable results.")
  args = parser.parse_args()

  if not torch.cuda.is_available() or not args.device.startswith("cuda"):
    raise SystemExit("scripts/tune.py measures CUDA memory and requires a CUDA device, e.g. --device cuda:0")

  device = torch.device(args.device)
  torch.cuda.set_device(device)
  host_total = host_ram_bytes()
  cuda_report = cuda_device_report(device)
  print("Host RAM:", "unknown" if host_total is None else f"{gib(host_total):.2f} GiB", f"rss={gib(proc_rss_bytes()):.2f} GiB")
  print(
    "CUDA:",
    f"cuda:{cuda_report['index']}",
    cuda_report["name"],
    f"total={cuda_report['total_gib']:.2f} GiB",
    f"free={cuda_report['free_gib']:.2f} GiB",
    f"capability={cuda_report['capability']}",
  )
  print(f"Loading model={args.model} dtype={args.dtype} flash_attention={args.flash_attention}")
  model, vocab_size = load_model(args, device)
  param_count = sum(param.numel() for param in model.parameters())
  print(
    f"Model loaded: params={param_count:,} vocab={vocab_size:,} "
    f"allocated={gib(torch.cuda.memory_allocated(device)):.2f} GiB "
    f"reserved={gib(torch.cuda.memory_reserved(device)):.2f} GiB"
  )

  results = []
  for optimizer_name in args.optimizers:
    for seq_len in args.seq_lens:
      for batch_size in args.batch_sizes:
        print(f"Running optimizer={optimizer_name} batch_size={batch_size} seq_len={seq_len}", flush=True)
        result = run_trial(model, vocab_size, optimizer_name, batch_size, seq_len, device, args)
        results.append(result)
        print_table([result])

  print()
  print_table(results)
  if args.json is not None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "host_total_gib": gib(host_total),
      "cuda": cuda_report,
      "model": args.model,
      "param_count": param_count,
      "vocab_size": vocab_size,
      "results": [asdict(result) for result in results],
    }
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.json}")


if __name__ == "__main__":
  main()
