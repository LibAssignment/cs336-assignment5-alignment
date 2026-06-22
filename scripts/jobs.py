from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cs336_alignment.rl.jobs import find_job_dir, job_summary, list_jobs


def format_duration(seconds: float | None) -> str:
  if seconds is None:
    return ""
  total = int(seconds)
  hours, remainder = divmod(total, 3600)
  minutes, secs = divmod(remainder, 60)
  if hours:
    return f"{hours}h{minutes:02d}m{secs:02d}s"
  if minutes:
    return f"{minutes}m{secs:02d}s"
  return f"{secs}s"


def format_steps(row: dict) -> str:
  latest = row.get("latest_step")
  target = row.get("target_steps")
  if latest is None and target is None:
    return ""
  if latest is None:
    latest = 0
  if target is None:
    return str(latest)
  return f"{latest}/{target}"


def print_table(rows: list[dict]) -> None:
  if not rows:
    print("no jobs")
    return
  print(f"{'run_id':>6}  {'status':<10}  {'step':>6}  {'duration':>10}  {'pid':>8}  path")
  for row in rows:
    pid = "" if row["pid"] is None else str(row["pid"])
    step = format_steps(row)
    duration = format_duration(row["duration_seconds"])
    print(f"{row['run_id']:>6}  {row['status']:<10}  {step:>6}  {duration:>10}  {pid:>8}  {row['path']}")


def cmd_list(args: argparse.Namespace) -> None:
  rows = list_jobs(args.job_root)
  if args.json:
    print(json.dumps(rows, indent=2, sort_keys=True))
  else:
    print_table(rows)


def cmd_status(args: argparse.Namespace) -> None:
  path = find_job_dir(args.job_root, args.run_id)
  if path is None:
    raise SystemExit(f"Job {args.run_id} not found under {args.job_root}")
  row = job_summary(path)
  if args.json:
    print(json.dumps(row, indent=2, sort_keys=True))
  else:
    print_table([row])


def main() -> None:
  parser = argparse.ArgumentParser(description="Inspect local GRPO training jobs.")
  parser.add_argument("--job-root", type=Path, default=Path("out/jobs"))
  subparsers = parser.add_subparsers(dest="command")

  list_parser = subparsers.add_parser("list")
  list_parser.add_argument("--json", action="store_true")
  list_parser.set_defaults(func=cmd_list)

  status_parser = subparsers.add_parser("status")
  status_parser.add_argument("run_id", type=int)
  status_parser.add_argument("--json", action="store_true")
  status_parser.set_defaults(func=cmd_status)

  args = parser.parse_args()
  if not hasattr(args, "func"):
    args.func = cmd_list
    args.json = False
  args.func(args)


if __name__ == "__main__":
  main()
