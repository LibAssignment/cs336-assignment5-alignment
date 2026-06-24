from __future__ import annotations

import argparse
import pickle
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BYTES_PER_GIB = 1024**3
DEFAULT_TUNE_ROOT = Path("out/tune")
CLASS_NAME_CACHE: dict[tuple[str, int], str | None] = {}


class Tee:
  def __init__(self, *files: TextIO) -> None:
    self.files = files

  def write(self, text: str) -> int:
    for file in self.files:
      file.write(text)
    return len(text)

  def flush(self) -> None:
    for file in self.files:
      file.flush()


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


@dataclass(frozen=True)
class TunePlotRow:
  optimizer: str
  batch_size: int
  seq_len: int
  status: str
  cuda_peak_allocated_gib: float

  @property
  def batch_tokens(self) -> int:
    return self.batch_size * self.seq_len


def gib(num_bytes: int | float | None) -> float | None:
  if num_bytes is None:
    return None
  return num_bytes / BYTES_PER_GIB


def format_gib(num_bytes: int | float | None) -> str:
  if num_bytes is None:
    return ""
  return f"{gib(num_bytes):.2f} GiB"


def parse_float(value: Any) -> float | None:
  if value is None or value == "":
    return None
  return float(value)


def parse_int_list(raw: str) -> list[int]:
  values = []
  for item in raw.split(","):
    item = item.strip()
    if item:
      values.append(int(item))
  if not values:
    raise argparse.ArgumentTypeError("expected at least one integer")
  return values


def parse_profile_trials(raw: str) -> set[tuple[str | None, int, int]]:
  specs = set()
  for item in raw.split(","):
    item = item.strip()
    if not item:
      continue
    match = re.fullmatch(r"(?:(\w+):)?(\d+)x(\d+)", item)
    if match is None:
      raise argparse.ArgumentTypeError("profile trials must look like 8x1024 or adamw:8x1024")
    optimizer, batch_size, seq_len = match.groups()
    specs.add((optimizer, int(batch_size), int(seq_len)))
  return specs


def slug_part(value: str) -> str:
  value = re.sub(r"[^A-Za-z0-9_.=-]+", "-", value.strip())
  return value.strip("-") or "default"


def list_slug(values: list[int], prefix: str) -> str:
  if not values:
    return f"{prefix}none"
  return prefix + "-".join(str(value) for value in values)


def default_run_name(args: argparse.Namespace) -> str:
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  parts = [
    timestamp,
    slug_part(args.model),
    slug_part(args.dtype),
    slug_part("-".join(args.optimizers)),
    list_slug(args.batch_sizes, "B"),
    list_slug(args.seq_lens, "S"),
  ]
  if args.include_entropy:
    parts.append("entropy")
  return "_".join(parts)


def resolve_run_grid(args: argparse.Namespace) -> None:
  if args.focus_profile_trials and args.profile_trials:
    profile_optimizers = sorted({optimizer for optimizer, _, _ in args.profile_trials if optimizer is not None})
    if args.optimizers is None and profile_optimizers:
      args.optimizers = profile_optimizers
    if args.batch_sizes is None:
      args.batch_sizes = sorted({batch_size for _, batch_size, _ in args.profile_trials})
    if args.seq_lens is None:
      args.seq_lens = sorted({seq_len for _, _, seq_len in args.profile_trials})

  if args.optimizers is None:
    args.optimizers = ["adamw"]
  if args.batch_sizes is None:
    args.batch_sizes = parse_int_list("1,2,4,8,16")
  if args.seq_lens is None:
    args.seq_lens = parse_int_list("256,512,768,1024,2048,4096")


def resolve_run_paths(args: argparse.Namespace) -> None:
  resolve_run_grid(args)
  run_name = args.name or default_run_name(args)
  args.run_dir = args.out_root / run_name
  args.output = args.output or args.run_dir / "tune.txt"
  if args.json is True:
    args.json = args.run_dir / "tune.json"
  args.profile_dir = args.profile_dir or args.run_dir / "profiles"


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
    from cs336_alignment.rl.models import tiny_byte_tokenizer, tiny_train_model

    tokenizer = tiny_byte_tokenizer()
    model = tiny_train_model(tokenizer, n_positions=max(args.seq_lens), device=device)
    return model, len(tokenizer)

  from transformers import AutoModelForCausalLM

  from cs336_alignment.rl.config import DEFAULT_OLMO2_1B_PATH

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


@contextmanager
def nvtx_range(name: str, enabled: bool):
  if not enabled:
    yield
    return
  torch.cuda.nvtx.range_push(name)
  try:
    yield
  finally:
    torch.cuda.nvtx.range_pop()


def should_profile_trial(args: argparse.Namespace, optimizer_name: str, batch_size: int, seq_len: int) -> bool:
  return any(
    (optimizer is None or optimizer == optimizer_name) and profile_batch_size == batch_size and profile_seq_len == seq_len
    for optimizer, profile_batch_size, profile_seq_len in args.profile_trials
  )


def memory_snapshot_path(args: argparse.Namespace, optimizer_name: str, batch_size: int, seq_len: int) -> Path:
  return args.profile_dir / f"memory_{optimizer_name}_B{batch_size}_S{seq_len}.pickle"


def start_memory_history(args: argparse.Namespace) -> None:
  record = torch.cuda.memory._record_memory_history
  try:
    record(enabled="all", context="all", stacks="all", max_entries=args.memory_history_max_entries)
  except TypeError:
    try:
      record(enabled=True, max_entries=args.memory_history_max_entries)
    except TypeError:
      record(max_entries=args.memory_history_max_entries)


def stop_memory_history() -> None:
  try:
    torch.cuda.memory._record_memory_history(enabled=None)
  except TypeError:
    torch.cuda.memory._record_memory_history(False)


def dump_memory_snapshot(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.cuda.memory._dump_snapshot(str(path))


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
    with nvtx_range(f"batch B{batch_size} S{seq_len}", args.nvtx):
      input_ids, labels = random_batch(batch_size, seq_len, vocab_size, device)
      attention_mask = torch.ones_like(input_ids)
    with nvtx_range(f"forward B{batch_size} S{seq_len}", args.nvtx):
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = outputs.logits
    with nvtx_range(f"loss B{batch_size} S{seq_len}", args.nvtx):
      loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="mean",
      )
      if args.include_entropy:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        loss = loss + args.entropy_weight * entropy
    with nvtx_range(f"backward B{batch_size} S{seq_len}", args.nvtx):
      loss.backward()
    with nvtx_range(f"optimizer {optimizer_name} B{batch_size} S{seq_len}", args.nvtx):
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


def print_table(results: list[TrialResult], *, show_errors: bool = False) -> None:
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
    if show_errors and row.error:
      print(f"  error: {row.error}")


def load_json_plot_rows(path: Path) -> list[TunePlotRow]:
  payload = json.loads(path.read_text())
  if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
    raise ValueError(f"{path} must contain an object with a results list")

  rows = []
  for item in payload["results"]:
    if not isinstance(item, dict):
      continue
    peak_allocated = parse_float(item.get("cuda_peak_allocated_gib"))
    if peak_allocated is None:
      continue
    rows.append(
      TunePlotRow(
        optimizer=str(item["optimizer"]),
        batch_size=int(item["batch_size"]),
        seq_len=int(item["seq_len"]),
        status=str(item["status"]),
        cuda_peak_allocated_gib=peak_allocated,
      )
    )
  return rows


def parse_text_plot_row(line: str) -> TunePlotRow | None:
  match = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(.*)$", line)
  if match is None:
    return None

  optimizer, batch_size, seq_len, status, rest = match.groups()
  if optimizer in {"opt", "Running", "Host", "CUDA:", "Model"}:
    return None

  values = rest.split()
  if len(values) == 8:
    peak_allocated_index = 2
  elif len(values) == 7:
    peak_allocated_index = 1
  else:
    return None

  peak_allocated = parse_float(values[peak_allocated_index])
  if peak_allocated is None:
    return None
  return TunePlotRow(
    optimizer=optimizer,
    batch_size=int(batch_size),
    seq_len=int(seq_len),
    status=status,
    cuda_peak_allocated_gib=peak_allocated,
  )


def load_text_plot_rows(path: Path) -> list[TunePlotRow]:
  rows = []
  for line in path.read_text().splitlines():
    row = parse_text_plot_row(line)
    if row is not None:
      rows.append(row)
  return rows


def dedupe_plot_rows(rows: list[TunePlotRow]) -> list[TunePlotRow]:
  deduped = {}
  for row in rows:
    deduped[(row.optimizer, row.batch_size, row.seq_len)] = row
  return list(deduped.values())


def load_plot_rows(path: Path) -> list[TunePlotRow]:
  if path.suffix == ".json":
    return dedupe_plot_rows(load_json_plot_rows(path))
  if path.suffix in {".txt", ".log", ".out"}:
    return dedupe_plot_rows(load_text_plot_rows(path))

  try:
    return dedupe_plot_rows(load_json_plot_rows(path))
  except (json.JSONDecodeError, ValueError, KeyError, TypeError):
    return dedupe_plot_rows(load_text_plot_rows(path))


def plot_memory(rows: list[TunePlotRow], path: Path) -> None:
  if not rows:
    raise SystemExit("no tune rows with peak memory values found")

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  markers = {"ok": "o", "oom": "x", "error": "s"}
  fig, ax = plt.subplots(figsize=(9, 5.5))
  for optimizer in sorted({row.optimizer for row in rows}):
    for status in ("ok", "oom", "error"):
      group = [row for row in rows if row.optimizer == optimizer and row.status == status]
      if not group:
        continue
      group.sort(key=lambda row: row.batch_tokens)
      ax.plot(
        [row.batch_tokens for row in group],
        [row.cuda_peak_allocated_gib for row in group],
        linestyle="-" if status == "ok" else "None",
        marker=markers.get(status, "."),
        label=f"{optimizer} {status}",
      )

  ax.set_xlabel("Batch tokens (B*S)")
  ax.set_ylabel("CUDA peak allocated (GiB)")
  ax.set_title("Tune Memory by Batch Tokens")
  ax.grid(True, which="both", linestyle=":", linewidth=0.7)
  ax.legend()
  fig.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=160)
  plt.close(fig)


def load_memory_snapshot(path: Path) -> dict[str, Any]:
  with path.open("rb") as snapshot_file:
    snapshot = pickle.load(snapshot_file)
  if not isinstance(snapshot, dict):
    raise ValueError(f"{path} did not contain a torch.cuda.memory snapshot dict")
  return snapshot


def flatten_device_traces(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
  traces = []
  for device_trace in snapshot.get("device_traces", []):
    if isinstance(device_trace, list):
      traces.extend(event for event in device_trace if isinstance(event, dict))
  return traces


def infer_class_name(filename: str, line: int) -> str | None:
  if filename == "??" or line <= 0:
    return None
  key = (filename, line)
  if key in CLASS_NAME_CACHE:
    return CLASS_NAME_CACHE[key]

  path = Path(filename)
  try:
    lines = path.read_text(errors="ignore").splitlines()
  except OSError:
    CLASS_NAME_CACHE[key] = None
    return None

  class_name = None
  class_indent = None
  for raw_line in lines[:line]:
    stripped = raw_line.lstrip()
    indent = len(raw_line) - len(stripped)
    if class_indent is not None and indent <= class_indent and stripped and not stripped.startswith(("#", "@")):
      class_name = None
      class_indent = None
    match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\b", stripped)
    if match is not None:
      class_name = match.group(1)
      class_indent = indent

  CLASS_NAME_CACHE[key] = class_name
  return class_name


def format_frame_name(filename: str, line: int, name: str) -> str:
  class_name = infer_class_name(filename, line)
  if class_name is not None and "." not in name:
    return f"{class_name}.{name}"
  return name


def frame_label(frames: list[dict[str, Any]]) -> str:
  skip_names = {"<module>", "main", "cmd_run", "cmd_profile", "run_tune"}
  for frame in frames:
    filename = str(frame.get("filename", ""))
    name = str(frame.get("name", ""))
    if filename == "??" or "site-packages/torch" in filename or name in skip_names:
      continue
    try:
      filename = str(Path(filename).resolve().relative_to(REPO_ROOT))
    except ValueError:
      pass
    line = frame.get("line", 0)
    return f"{filename}:{line} {format_frame_name(str(frame.get('filename', '')), int(line), name)}"
  for frame in frames:
    filename = str(frame.get("filename", ""))
    name = str(frame.get("name", ""))
    if filename != "??" and name not in skip_names:
      line = int(frame.get("line", 0))
      return f"{filename}:{line} {format_frame_name(filename, line, name)}"
  return "<unknown>"


def allocation_group(site: str, size: int, *, split_size: bool) -> str:
  if not split_size:
    return site
  return f"{site} [alloc_size={format_gib(size)}]"


def add_group_stats(groups: dict[str, dict[str, int]], group: str, size: int) -> None:
  groups[group]["count"] += 1
  groups[group]["bytes"] += size


def print_group_table(title: str, groups: dict[str, dict[str, int]], *, top: int) -> None:
  print(title)
  if not groups:
    print("  <none>")
    return
  for group, stats in sorted(groups.items(), key=lambda item: item[1]["bytes"], reverse=True)[:top]:
    print(f"  {format_gib(stats['bytes']):>10} {stats['count']:>6}x  {group}")


def print_memory_snapshot_analysis(path: Path, *, top: int, split_size: bool) -> None:
  snapshot = load_memory_snapshot(path)
  segments = [segment for segment in snapshot.get("segments", []) if isinstance(segment, dict)]
  traces = flatten_device_traces(snapshot)

  total_segment_bytes = sum(int(segment.get("total_size", 0)) for segment in segments)
  active_segment_bytes = sum(int(segment.get("active_size", 0)) for segment in segments)
  allocated_segment_bytes = sum(int(segment.get("allocated_size", 0)) for segment in segments)
  requested_segment_bytes = sum(int(segment.get("requested_size", 0)) for segment in segments)
  inactive_segment_bytes = total_segment_bytes - active_segment_bytes

  blocks = []
  for segment in segments:
    for block in segment.get("blocks", []):
      if isinstance(block, dict):
        blocks.append(block)
  block_state_counts = Counter(str(block.get("state", "unknown")) for block in blocks)
  block_state_bytes = defaultdict(int)
  for block in blocks:
    block_state_bytes[str(block.get("state", "unknown"))] += int(block.get("size", 0))

  actions = Counter(str(event.get("action", "unknown")) for event in traces)
  live_bytes = 0
  reserved_bytes = 0
  peak_live_bytes = 0
  peak_reserved_bytes = 0
  live_allocations: dict[int, tuple[int, str, str]] = {}
  peak_live_allocations: dict[int, tuple[int, str, str]] = {}
  allocation_sites: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
  largest_events = []

  for event in sorted(traces, key=lambda item: int(item.get("time_us", 0))):
    action = event.get("action")
    addr = event.get("addr")
    size = int(event.get("size", 0) or 0)
    if action == "segment_alloc":
      reserved_bytes += size
      peak_reserved_bytes = max(peak_reserved_bytes, reserved_bytes)
    elif action == "segment_free":
      reserved_bytes = max(0, reserved_bytes - size)
    elif action == "alloc":
      site = frame_label(event.get("frames", []))
      group = allocation_group(site, size, split_size=split_size)
      live_allocations[int(addr)] = (size, site, group)
      live_bytes += size
      if live_bytes > peak_live_bytes:
        peak_live_bytes = live_bytes
        peak_live_allocations = dict(live_allocations)
      add_group_stats(allocation_sites, group, size)
      largest_events.append((size, site))
    elif action == "free_requested":
      allocation = live_allocations.pop(int(addr), None)
      live_bytes = max(0, live_bytes - (allocation[0] if allocation is not None else size))

  peak_live_sites: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
  for size, _, group in peak_live_allocations.values():
    add_group_stats(peak_live_sites, group, size)

  print(f"Snapshot: {path}")
  print()
  print("Segments at snapshot dump")
  print(f"  segments:          {len(segments)}")
  print(f"  total reserved:    {format_gib(total_segment_bytes)}")
  print(f"  active:            {format_gib(active_segment_bytes)}")
  print(f"  allocated:         {format_gib(allocated_segment_bytes)}")
  print(f"  requested:         {format_gib(requested_segment_bytes)}")
  print(f"  inactive/fragment: {format_gib(inactive_segment_bytes)}")
  print()
  print("Trace peaks")
  print(f"  peak live allocations: {format_gib(peak_live_bytes)}")
  print(f"  peak reserved memory:  {format_gib(peak_reserved_bytes)}")
  print()
  print("Event counts")
  for action, count in actions.most_common():
    print(f"  {action:<16} {count:>8}")
  print()
  print("Block states at snapshot dump")
  for state, count in block_state_counts.most_common():
    print(f"  {state:<18} {count:>8} {format_gib(block_state_bytes[state]):>12}")
  print()
  print_group_table(f"Top {top} allocation sites by total allocated bytes", allocation_sites, top=top)
  print()
  print_group_table(f"Top {top} live allocation sites at peak memory", peak_live_sites, top=top)
  print()
  print(f"Top {top} single allocation events")
  for size, site in sorted(largest_events, reverse=True)[:top]:
    print(f"  {format_gib(size):>10}  {site}")


def run_tune(args: argparse.Namespace) -> None:
  global torch

  import torch

  if args.memory_history and not args.profile_trials:
    raise SystemExit("--memory-history requires --profile-trials, e.g. --profile-trials 8x1024")
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
        profile_trial = should_profile_trial(args, optimizer_name, batch_size, seq_len)
        snapshot_path = memory_snapshot_path(args, optimizer_name, batch_size, seq_len)
        record_memory_history = args.memory_history and profile_trial
        if record_memory_history:
          print(f"recording torch.cuda.memory history for {optimizer_name} B={batch_size} S={seq_len}", flush=True)
          start_memory_history(args)
        try:
          result = run_trial(model, vocab_size, optimizer_name, batch_size, seq_len, device, args)
          if record_memory_history:
            dump_memory_snapshot(snapshot_path)
            print(f"wrote {snapshot_path}", flush=True)
        finally:
          if record_memory_history:
            stop_memory_history()
        results.append(result)
        print_table([result], show_errors=True)

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


def add_run_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--model", choices=["tiny", "olmo2-1B"], default="olmo2-1B")
  parser.add_argument("--model-path", type=Path, default=None)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--batch-sizes", type=parse_int_list, default=None)
  parser.add_argument("--seq-lens", type=parse_int_list, default=None)
  parser.add_argument("--optimizers", type=lambda raw: [item.strip() for item in raw.split(",") if item.strip()], default=None)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
  parser.add_argument("--flash-attention", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--include-entropy", action="store_true", help="Also compute the full-vocab entropy term used by the current GRPO helper.")
  parser.add_argument("--entropy-weight", type=float, default=0.0)
  parser.add_argument("--nvtx", action=argparse.BooleanOptionalAction, default=True, help="Emit NVTX ranges for Nsight Systems.")
  parser.add_argument("--out-root", type=Path, default=DEFAULT_TUNE_ROOT, help="Root directory for named tune runs.")
  parser.add_argument("--name", default=None, help="Run directory name. Defaults to timestamp plus key parameters.")
  parser.add_argument("--output", type=Path, default=None, help="Path to write the human-readable tune output. Defaults to <run-dir>/tune.txt.")
  parser.add_argument(
    "--json",
    type=Path,
    nargs="?",
    const=True,
    default=None,
    help="Optional path to write machine-readable results. Defaults to <run-dir>/tune.json when passed without a path.",
  )


def add_profile_args(parser: argparse.ArgumentParser) -> None:
  add_run_args(parser)
  parser.add_argument("trials", nargs="?", default=None, help="Trials to profile, e.g. 8x1024 or adamw:8x1024,16x512.")
  parser.add_argument("--profile-trials", type=parse_profile_trials, default=None, help="Comma-separated trials to profile, e.g. 8x1024 or adamw:8x1024.")
  parser.add_argument("--profile-dir", type=Path, default=None, help="Profile output directory. Defaults to <run-dir>/profiles.")
  parser.add_argument("--memory-history", action=argparse.BooleanOptionalAction, default=True, help="Dump torch.cuda.memory snapshots.")
  parser.add_argument("--memory-history-max-entries", type=int, default=1_000_000)


def tune_input_in_dir(path: Path) -> Path:
  json_path = path / "tune.json"
  if json_path.exists():
    return json_path
  return path / "tune.txt"


def default_plot_input_path() -> Path:
  if not DEFAULT_TUNE_ROOT.exists():
    return tune_input_in_dir(DEFAULT_TUNE_ROOT / "latest")
  run_dirs = [path for path in DEFAULT_TUNE_ROOT.iterdir() if path.is_dir()]
  if not run_dirs:
    return tune_input_in_dir(DEFAULT_TUNE_ROOT / "latest")
  return tune_input_in_dir(max(run_dirs, key=lambda path: path.stat().st_mtime))


def cmd_run(args: argparse.Namespace) -> None:
  args.profile_trials = set()
  args.profile_dir = None
  args.memory_history = False
  args.memory_history_max_entries = 1_000_000
  args.focus_profile_trials = False
  resolve_run_paths(args)
  args.run_dir.mkdir(parents=True, exist_ok=True)
  print(f"tune output: {args.run_dir}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w") as output_file:
    with redirect_stdout(Tee(sys.stdout, output_file)):
      print(f"tune output: {args.run_dir}")
      run_tune(args)
      print(f"wrote {args.output}")


def cmd_profile(args: argparse.Namespace) -> None:
  if args.profile_trials is None:
    if args.trials is None:
      raise SystemExit("profile requires trials, e.g. python scripts/tune.py profile 8x1024")
    args.profile_trials = parse_profile_trials(args.trials)
  if args.json is None:
    args.json = True
  args.focus_profile_trials = True
  resolve_run_paths(args)
  args.run_dir.mkdir(parents=True, exist_ok=True)
  print(f"profile output: {args.run_dir}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("w") as output_file:
    with redirect_stdout(Tee(sys.stdout, output_file)):
      print(f"profile output: {args.run_dir}")
      run_tune(args)
      print(f"wrote {args.output}")


def cmd_plot(args: argparse.Namespace) -> None:
  input_path = tune_input_in_dir(args.input) if args.input.is_dir() else args.input
  output_path = args.output or input_path.with_suffix(".png")
  rows = load_plot_rows(input_path)
  plot_memory(rows, output_path)
  print(f"read {len(rows)} rows from {input_path}")
  print(f"wrote {output_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
  print_memory_snapshot_analysis(args.snapshot, top=args.top, split_size=args.split_size)


def main() -> None:
  parser = argparse.ArgumentParser(description="Measure or plot tune memory results.")
  subparsers = parser.add_subparsers(dest="command")

  run_parser = subparsers.add_parser("run", help="measure CUDA forward/backward memory")
  add_run_args(run_parser)
  run_parser.set_defaults(func=cmd_run)

  profile_parser = subparsers.add_parser("profile", help="profile selected trials with torch.cuda.memory history")
  add_profile_args(profile_parser)
  profile_parser.set_defaults(func=cmd_profile)

  plot_parser = subparsers.add_parser("plot", help="plot B*S versus peak memory from tune txt/json")
  plot_parser.add_argument("input", nargs="?", type=Path, default=default_plot_input_path(), help="Input tune txt/json file.")
  plot_parser.add_argument("--output", type=Path, default=None, help="Output plot path. Defaults to a .png sibling of the input.")
  plot_parser.set_defaults(func=cmd_plot)

  analyze_parser = subparsers.add_parser("analyze", help="summarize a torch.cuda.memory snapshot pickle")
  analyze_parser.add_argument("snapshot", type=Path, help="Path to memory_*.pickle from profile output.")
  analyze_parser.add_argument("--top", type=int, default=12, help="Number of top allocation sites/events to show.")
  analyze_parser.add_argument("--split-size", action="store_true", help="Treat same-frame allocations with different sizes as separate groups.")
  analyze_parser.set_defaults(func=cmd_analyze)

  add_run_args(parser)
  parser.set_defaults(func=cmd_run)
  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
