from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cs336_alignment.rl.jobs import Job, clear_pid, find_job_dir, job_summary, list_jobs, process_is_alive, read_json, write_status


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


def require_job(job_root: Path, run_id: int) -> Job:
  path = find_job_dir(job_root, run_id)
  if path is None:
    raise SystemExit(f"Job {run_id} not found under {job_root}")
  return Job(run_id=run_id, path=path)


def terminate_process_group(pid: int, timeout: float) -> str:
  if not process_is_alive(pid):
    return "not running"
  try:
    os.killpg(pid, signal.SIGTERM)
  except ProcessLookupError:
    return "not running"
  except OSError:
    os.kill(pid, signal.SIGTERM)

  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if not process_is_alive(pid):
      return "terminated"
    time.sleep(0.2)

  try:
    os.killpg(pid, signal.SIGKILL)
  except ProcessLookupError:
    return "terminated"
  except OSError:
    os.kill(pid, signal.SIGKILL)
  return "killed"


def cmd_stop(args: argparse.Namespace) -> None:
  job = require_job(args.job_root, args.run_id)
  row = job_summary(job.path)
  if row["status"] in {"succeeded", "failed"}:
    print(f"job {job.run_id} already {row['status']}")
    return
  pid = row["pid"]
  if pid is None:
    print(f"job {job.run_id} has no pid; status unchanged ({row['status']})")
    return

  result = terminate_process_group(pid, args.timeout)
  clear_pid(job)
  write_status(job, "stopped", stopped_pid=pid, stop_result=result)
  print(f"job {job.run_id} stopped: {result} pid={pid}")


def resume_argv(job_root: Path, job: Job) -> list[str]:
  job_config = read_json(job.path / "job_config.json")
  argv = [
    sys.executable,
    str(REPO_ROOT / "scripts" / "train.py"),
    "--resume",
    "--run-id",
    str(job.run_id),
    "--job-root",
    str(job_root),
    "--config",
    str(job.path / "train_config.json"),
  ]
  if (job.path / "wandb_config.json").exists():
    argv.extend(["--wandb-config", str(job.path / "wandb_config.json")])
  checkpoint_every = job_config.get("checkpoint_every")
  if checkpoint_every is not None:
    argv.extend(["--checkpoint-every", str(checkpoint_every)])
  return argv


def cmd_resume(args: argparse.Namespace) -> None:
  job = require_job(args.job_root, args.run_id)
  row = job_summary(job.path)
  if row["status"] == "running" and row["pid"] is not None and process_is_alive(row["pid"]):
    raise SystemExit(f"Job {job.run_id} is already running with pid {row['pid']}")

  with job.stdout_path.open("ab") as stdout_file, job.stderr_path.open("ab") as stderr_file:
    process = subprocess.Popen(
      resume_argv(args.job_root, job),
      stdout=stdout_file,
      stderr=stderr_file,
      stdin=subprocess.DEVNULL,
      start_new_session=True,
    )
  write_status(job, "queued", launcher_pid=os.getpid(), child_pid=process.pid, resumed=True)
  print(f"resumed job {job.run_id}: {job.path}")


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

  stop_parser = subparsers.add_parser("stop")
  stop_parser.add_argument("run_id", type=int)
  stop_parser.add_argument("--timeout", type=float, default=10.0)
  stop_parser.set_defaults(func=cmd_stop)

  resume_parser = subparsers.add_parser("resume")
  resume_parser.add_argument("run_id", type=int)
  resume_parser.set_defaults(func=cmd_resume)

  args = parser.parse_args()
  if not hasattr(args, "func"):
    args.func = cmd_list
    args.json = False
  args.func(args)


if __name__ == "__main__":
  main()
