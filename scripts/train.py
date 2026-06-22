from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cs336_alignment.rl.config import parse_train_config
from cs336_alignment.rl.jobs import prepare_job, write_status
from cs336_alignment.rl.train import train


def background_argv(run_id: int) -> list[str]:
  args = [arg for arg in sys.argv[1:] if arg != "--job"]
  if "--run-id" not in args:
    args.extend(["--run-id", str(run_id)])
  return [sys.executable, str(Path(__file__).resolve()), *args]


def launch_background(parsed_config) -> None:
  if not parsed_config.job.enabled:
    raise SystemExit("--job cannot be combined with --no-job")
  job = prepare_job(parsed_config.job, parsed_config.train, parsed_config.wandb)
  write_status(job, "queued")
  with job.stdout_path.open("ab") as stdout_file, job.stderr_path.open("ab") as stderr_file:
    process = subprocess.Popen(
      background_argv(job.run_id),
      stdout=stdout_file,
      stderr=stderr_file,
      stdin=subprocess.DEVNULL,
      start_new_session=True,
    )
  write_status(job, "queued", launcher_pid=os.getpid(), child_pid=process.pid)
  print(f"started job {job.run_id}: {job.path}")


def main() -> None:
  parsed_config = parse_train_config()
  if parsed_config.job.background:
    launch_background(parsed_config)
    return
  job_config = parsed_config.job if parsed_config.job.enabled else None
  train(parsed_config.train, parsed_config.wandb, job_config)


if __name__ == "__main__":
  main()
