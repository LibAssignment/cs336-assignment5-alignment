from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


EXACT_R1_BOUNDARY = "</think> <answer>"
LOCAL_JOB_ROOT = Path("out/jobs")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  with path.open() as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def load_metric_by_step(job: Path, metric: str) -> dict[int, float]:
  path = job / "metrics.jsonl"
  if not path.exists():
    return {}
  result = {}
  for row in load_jsonl(path):
    if metric in row:
      result[int(row["step"])] = float(row[metric])
  return result


def rollout_step(path: Path) -> int:
  match = re.search(r"_step_(\d+)\.jsonl$", path.name)
  if match is None:
    raise ValueError(f"Could not parse rollout step from {path}")
  return int(match.group(1))


def job_run_id(path: Path) -> int | None:
  match = re.search(r"-(\d+)$", path.name)
  return int(match.group(1)) if match is not None else None


def find_job_by_run_id(job_root: Path, run_id: int) -> list[Path]:
  if not job_root.exists():
    return []
  return sorted(path for path in job_root.iterdir() if path.is_dir() and job_run_id(path) == run_id)


def resolve_job(job_spec: str) -> Path:
  path = Path(job_spec)
  if path.exists():
    return path

  if ":" in job_spec:
    host, raw_run_id = job_spec.split(":", maxsplit=1)
    if not raw_run_id.isdigit():
      raise SystemExit(f"Invalid job id {job_spec!r}; expected HOST:RUN_ID")
    run_id = int(raw_run_id)
    roots = [Path("results") / host / "jobs", Path("out") / host / "jobs"]
    matches = [match for root in roots for match in find_job_by_run_id(root, run_id)]
  elif job_spec.isdigit():
    run_id = int(job_spec)
    matches = find_job_by_run_id(LOCAL_JOB_ROOT, run_id)
  else:
    raise SystemExit(f"Job {job_spec!r} is not a path, RUN_ID, or HOST:RUN_ID")

  if len(matches) == 1:
    return matches[0]
  if not matches:
    raise SystemExit(f"No job found for {job_spec!r}")
  choices = "\n".join(f"  {match}" for match in matches)
  raise SystemExit(f"Job {job_spec!r} is ambiguous; use a path or HOST:RUN_ID:\n{choices}")


def sample_paths(job: Path, split: str) -> list[Path]:
  paths = sorted((job / "samples").glob(f"rollout_{split}_step_*.jsonl"), key=rollout_step)
  if not paths:
    raise SystemExit(f"No rollout_{split}_step_*.jsonl files found under {job / 'samples'}")
  return paths


def is_exact_r1_format(output: str) -> bool:
  return EXACT_R1_BOUNDARY in output and "</answer>" in output


def classify_output(output: str, answer: str | None) -> str:
  has_think_close = "</think>" in output
  has_answer_open = "<answer>" in output
  has_answer_close = "</answer>" in output
  has_exact_boundary = EXACT_R1_BOUNDARY in output

  if has_exact_boundary and has_answer_close:
    if has_contamination(output):
      return "format_ok_but_contaminated"
    return "format_ok"
  if not has_think_close and not has_answer_open:
    return "missing_think_close_and_answer"
  if not has_think_close:
    return "missing_think_close"
  if not has_answer_open:
    return "missing_answer_open"
  if not has_answer_close:
    return "missing_answer_close"
  if has_think_close and has_answer_open and not has_exact_boundary:
    return "bad_think_answer_boundary"
  if answer is None:
    return "unextractable_answer"
  return "other_format_failure"


def has_contamination(output: str) -> bool:
  first_answer_close = output.find("</answer>")
  tail = output[first_answer_close + len("</answer>") :] if first_answer_close >= 0 else output
  markers = ["\nUser:", "\nAssistant:", "\nQ:", "\n[Q]", "\nQuestion:", "\nA:"]
  return any(marker in tail for marker in markers)


def row_features(row: dict[str, Any]) -> dict[str, Any]:
  output = str(row.get("output", ""))
  answer = row.get("answer")
  exact_format = is_exact_r1_format(output)
  return {
    "format_ok": exact_format,
    "answer_extracted": answer is not None,
    "class": classify_output(output, answer if isinstance(answer, str) else None),
    "output_chars": len(output),
    "answer_chars": len(answer) if isinstance(answer, str) else 0,
    "has_contamination": has_contamination(output),
    "think_close_count": output.count("</think>"),
    "answer_open_count": output.count("<answer>"),
    "answer_close_count": output.count("</answer>"),
    "answer_correct": is_correct_answer(answer, row.get("ground_truth")),
  }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
  features = [row_features(row) for row in rows]
  classes = Counter(feature["class"] for feature in features)
  count = len(features)
  return {
    "count": count,
    "format_rate": mean(feature["format_ok"] for feature in features) if features else 0.0,
    "extract_rate": mean(feature["answer_extracted"] for feature in features) if features else 0.0,
    "correct_rate": mean(feature["answer_correct"] for feature in features) if features else 0.0,
    "contamination_rate": mean(feature["has_contamination"] for feature in features) if features else 0.0,
    "mean_output_chars": mean(feature["output_chars"] for feature in features) if features else 0.0,
    "classes": classes,
  }


def print_table(job: Path, split: str, paths: list[Path]) -> None:
  reward_metric = f"{split}/reward/mean" if split == "train" else f"{split}/reward/mean"
  format_metric = f"{split}/format_reward"
  answer_metric = f"{split}/answer_reward"
  reward_by_step = load_metric_by_step(job, reward_metric)
  format_by_step = load_metric_by_step(job, format_metric)
  answer_by_step = load_metric_by_step(job, answer_metric)
  print(f"job={job}")
  print(f"split={split} files={len(paths)}")
  print()
  print("step rows reward format answer sample_format extracted correct contam out_chars top_failures")
  for path in paths:
    step = rollout_step(path)
    rows = load_jsonl(path)
    summary = summarize_rows(rows)
    failures = Counter(summary["classes"])
    failures.pop("format_ok", None)
    failures.pop("format_ok_but_contaminated", None)
    top_failures = ",".join(f"{name}:{count}" for name, count in failures.most_common(3))
    reward_text = metric_text(reward_by_step.get(step))
    format_text = metric_text(format_by_step.get(step))
    answer_text = metric_text(answer_by_step.get(step))
    print(
      f"{step:>4} {summary['count']:>4} {reward_text:>6} {format_text:>6} {answer_text:>6} "
      f"{summary['format_rate']:.4f} {summary['extract_rate']:.4f} {summary['correct_rate']:.4f} "
      f"{summary['contamination_rate']:.4f} {summary['mean_output_chars']:.1f} "
      f"{top_failures}"
    )


def is_correct_answer(answer: Any, ground_truth: Any) -> bool:
  answer_value = normalize_numeric_answer(answer)
  ground_truth_value = normalize_numeric_answer(ground_truth)
  return answer_value is not None and ground_truth_value is not None and answer_value == ground_truth_value


def normalize_numeric_answer(value: Any) -> Decimal | None:
  if value is None:
    return None
  text = str(value).replace(",", "")
  matches = NUMBER_RE.findall(text)
  if not matches:
    return None
  try:
    return Decimal(matches[-1])
  except InvalidOperation:
    return None


def question_text(prompt: str, limit: int = 96) -> str:
  marker = "\nUser:"
  question = prompt.rsplit(marker, maxsplit=1)[-1] if marker in prompt else prompt
  question = question.split("\nAssistant:", maxsplit=1)[0].strip()
  question = re.sub(r"\s+", " ", question)
  if len(question) <= limit:
    return question
  return question[: limit - 3] + "..."


def print_answer_stability(paths: list[Path], max_examples: int) -> None:
  by_prompt: dict[str, dict[int, list[dict[str, Any]]]] = {}
  steps = [rollout_step(path) for path in paths]
  for path in paths:
    step = rollout_step(path)
    for row in load_jsonl(path):
      prompt = str(row.get("prompt", ""))
      by_prompt.setdefault(prompt, {}).setdefault(step, []).append(row)

  prompt_summaries = []
  for prompt, rows_by_step in by_prompt.items():
    rates = []
    answers_by_step = []
    ground_truth = None
    for step in steps:
      rows = rows_by_step.get(step, [])
      correct = [is_correct_answer(row.get("answer"), row.get("ground_truth")) for row in rows]
      rates.append(mean(correct) if correct else 0.0)
      answers_by_step.append(Counter(str(row.get("answer")) for row in rows).most_common(1)[0][0] if rows else "-")
      if rows and ground_truth is None:
        ground_truth = rows[0].get("ground_truth")
    first = rates[0] if rates else 0.0
    last = rates[-1] if rates else 0.0
    changed = len(set(rates)) > 1
    prompt_summaries.append(
      {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "rates": rates,
        "answers": answers_by_step,
        "first": first,
        "last": last,
        "changed": changed,
      }
    )

  total = len(prompt_summaries)
  always_correct = sum(all(rate == 1.0 for rate in item["rates"]) for item in prompt_summaries)
  always_wrong = sum(all(rate == 0.0 for rate in item["rates"]) for item in prompt_summaries)
  changed = sum(item["changed"] for item in prompt_summaries)
  improved = sum(item["last"] > item["first"] for item in prompt_summaries)
  regressed = sum(item["last"] < item["first"] for item in prompt_summaries)
  same_endpoint = total - improved - regressed

  print()
  print("answer_stability")
  print(f"steps={','.join(str(step) for step in steps)} prompts={total}")
  print(
    f"always_correct={always_correct} always_wrong={always_wrong} changed={changed} "
    f"improved={improved} regressed={regressed} same_endpoint={same_endpoint}"
  )

  if max_examples <= 0:
    return
  unstable = [item for item in prompt_summaries if item["changed"]]
  unstable.sort(key=lambda item: (item["last"] - item["first"], item["last"]))
  if not unstable:
    return
  print()
  print("unstable_answer_examples")
  for item in unstable[:max_examples]:
    trace = "".join(rate_marker(rate) for rate in item["rates"])
    print(f"  trace={trace} first={item['first']:.2f} last={item['last']:.2f} ground_truth={item['ground_truth']!r}")
    print(f"  question={question_text(item['prompt'])}")
    print(f"  answers={item['answers']}")


def rate_marker(rate: float) -> str:
  if rate == 1.0:
    return "1"
  if rate == 0.0:
    return "0"
  return "~"


def metric_text(value: float | None) -> str:
  return f"{value:.4f}" if value is not None else "-"


def select_paths(
  paths: list[Path],
  requested_steps: list[int] | None,
  every: int | None,
  first: int | None,
  last: int | None,
) -> list[Path]:
  selected = paths
  if requested_steps:
    requested = set(requested_steps)
    selected = [path for path in selected if rollout_step(path) in requested]
  if every is not None:
    selected = [path for index, path in enumerate(selected) if index % every == 0]
  if first is not None:
    selected = selected[:first]
  if last is not None:
    selected = selected[-last:]
  if not selected:
    raise SystemExit("No rollout files matched the requested selection.")
  return selected


def truncate(text: str, limit: int) -> str:
  text = text.replace("\r", "")
  if len(text) <= limit:
    return text
  return text[:limit] + "\n...<truncated>..."


def print_examples(paths: list[Path], examples_per_class: int, char_limit: int) -> None:
  if examples_per_class <= 0:
    return
  print()
  print("examples")
  for path in paths:
    step = rollout_step(path)
    rows = load_jsonl(path)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
      klass = row_features(row)["class"]
      if klass == "format_ok":
        continue
      buckets.setdefault(klass, []).append(row)
    print()
    print(f"step={step}")
    for klass, class_rows in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
      print(f"  {klass}: {len(class_rows)}")
      for row in class_rows[:examples_per_class]:
        print(f"    ground_truth={row.get('ground_truth')!r} answer={row.get('answer')!r}")
        print(indent(truncate(str(row.get("output", "")), char_limit), "    output: "))


def indent(text: str, prefix: str) -> str:
  return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Analyze saved rollout JSONL files for format failures.")
  parser.add_argument("job", help="Job path, RUN_ID, or HOST:RUN_ID, e.g. results/vastai3/jobs/20260625-051324-2 or vastai3:2.")
  parser.add_argument("--split", choices=["train", "validate"], default="train")
  parser.add_argument("--steps", type=lambda raw: [int(item) for item in raw.split(",") if item], default=None, help="Analyze only these comma-separated rollout steps.")
  parser.add_argument("--every", type=positive_int, default=None, help="Analyze every Nth selected rollout file.")
  parser.add_argument("--first", type=positive_int, default=None, help="Analyze only the first N selected rollout files.")
  parser.add_argument("--last", type=positive_int, default=None, help="Analyze only the last N selected rollout files.")
  parser.add_argument("--examples-last", type=positive_int, default=3, help="Show examples for the last N analyzed rollout files.")
  parser.add_argument("--examples-per-class", type=int, default=2)
  parser.add_argument("--answer-stability", action="store_true", help="Group repeated validation prompts and report whether answer correctness changes across selected steps.")
  parser.add_argument("--stability-examples", type=int, default=8, help="Maximum number of changed prompts to show in --answer-stability output.")
  parser.add_argument("--char-limit", type=int, default=900)
  return parser.parse_args()


def positive_int(raw: str) -> int:
  value = int(raw)
  if value <= 0:
    raise argparse.ArgumentTypeError("must be positive")
  return value


def main() -> None:
  args = parse_args()
  job = resolve_job(args.job)
  paths = sample_paths(job, args.split)
  selected_paths = select_paths(paths, args.steps, args.every, args.first, args.last)
  print_table(job, args.split, selected_paths)
  if args.answer_stability:
    print_answer_stability(selected_paths, args.stability_examples)
  print_examples(selected_paths[-args.examples_last :], args.examples_per_class, args.char_limit)


if __name__ == "__main__":
  main()
