from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import fcntl
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from .config import JobConfig, TrainConfig, WandbConfig


@dataclass(frozen=True)
class Job:
  run_id: int
  path: Path

  @property
  def checkpoints_dir(self) -> Path:
    return self.path / "checkpoints"

  @property
  def latest_path(self) -> Path:
    return self.path / "latest"

  @property
  def log_path(self) -> Path:
    return self.path / "train.log"

  @property
  def stdout_path(self) -> Path:
    return self.path / "stdout.log"

  @property
  def stderr_path(self) -> Path:
    return self.path / "stderr.log"

  @property
  def pid_path(self) -> Path:
    return self.path / "pid"

  @property
  def status_path(self) -> Path:
    return self.path / "status"

  @property
  def metadata_path(self) -> Path:
    return self.path / "job.json"


def now() -> float:
  return time.time()


def read_json(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  data = json.loads(path.read_text())
  if not isinstance(data, dict):
    raise ValueError(f"{path} must contain a JSON object")
  return data


def write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def job_dirs(job_root: Path) -> list[Path]:
  if not job_root.exists():
    return []
  return sorted(path for path in job_root.iterdir() if path.is_dir())


def run_id_from_dir(path: Path) -> int | None:
  metadata = read_json(path / "job.json")
  if "run_id" in metadata:
    return int(metadata["run_id"])
  if path.name.isdigit():
    return int(path.name)
  suffix = path.name.rsplit("-", 1)[-1]
  return int(suffix) if suffix.isdigit() else None


def existing_run_ids(job_root: Path) -> list[int]:
  run_ids = [run_id_from_dir(path) for path in job_dirs(job_root)]
  return sorted(run_id for run_id in run_ids if run_id is not None)


def job_dir_name(run_id: int) -> str:
  return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_id}"


@contextmanager
def job_root_lock(job_root: Path):
  job_root.mkdir(parents=True, exist_ok=True)
  lock_path = job_root / ".lock"
  with lock_path.open("w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
      yield
    finally:
      fcntl.flock(lock_file, fcntl.LOCK_UN)


def allocate_job_dir(job_root: Path) -> tuple[int, Path]:
  with job_root_lock(job_root):
    next_path = job_root / ".next_run_id"
    if next_path.exists():
      run_id = int(next_path.read_text().strip())
    else:
      existing = existing_run_ids(job_root)
      run_id = (max(existing) + 1) if existing else 1

    while True:
      job_path = job_root / job_dir_name(run_id)
      try:
        job_path.mkdir()
      except FileExistsError:
        run_id += 1
        continue
      next_path.write_text(f"{run_id + 1}\n")
      return run_id, job_path


def find_job_dir(job_root: Path, run_id: int) -> Path | None:
  for path in job_dirs(job_root):
    if run_id_from_dir(path) == run_id:
      return path
  return None


def prepare_job(job_config: JobConfig, train_config: TrainConfig, wandb_config: WandbConfig | None) -> Job:
  job_root = job_config.job_root
  if job_config.run_id is None:
    run_id, job_path = allocate_job_dir(job_root)
  else:
    run_id = job_config.run_id
    job_path = find_job_dir(job_root, run_id)
    if job_config.resume:
      if job_path is None:
        raise FileNotFoundError(f"Cannot resume missing job {run_id} under {job_root}")
    else:
      if job_path is not None:
        status = read_json(job_path / "job.json").get("status")
        if status not in {"created", "queued"}:
          raise FileExistsError(f"Job {run_id} already exists at {job_path}")
      else:
        job_path = job_root / job_dir_name(run_id)
        job_path.mkdir(parents=True, exist_ok=False)

  job = Job(run_id=run_id, path=job_path)
  job.path.mkdir(parents=True, exist_ok=True)
  job.checkpoints_dir.mkdir(parents=True, exist_ok=True)
  train_config.save_json(job.path / "train_config.json")
  if wandb_config is not None:
    write_json(job.path / "wandb_config.json", wandb_config.to_dict())
  write_json(job.path / "job_config.json", job_config.to_dict() | {"run_id": run_id})
  write_status(job, "created")
  return job


def write_status(job: Job, status: str, **updates: Any) -> None:
  metadata = read_json(job.metadata_path)
  created_at = metadata.get("created_at", now())
  metadata.update(
    {
      "run_id": job.run_id,
      "job_name": job.path.name,
      "status": status,
      "created_at": created_at,
      "updated_at": now(),
      **updates,
    }
  )
  write_json(job.metadata_path, metadata)
  job.status_path.write_text(status + "\n")


def write_pid(job: Job) -> None:
  job.pid_path.write_text(f"{os.getpid()}\n")


def clear_pid(job: Job) -> None:
  job.pid_path.unlink(missing_ok=True)


def checkpoint_dir(job: Job, step: int) -> Path:
  return job.checkpoints_dir / f"step_{step:06d}"


def write_latest(job: Job, path: Path) -> None:
  relative = path.relative_to(job.path)
  job.latest_path.write_text(str(relative) + "\n")


def latest_checkpoint(job: Job) -> Path | None:
  if not job.latest_path.exists():
    return None
  raw = job.latest_path.read_text().strip()
  if not raw:
    return None
  path = job.path / raw
  return path if path.exists() else None


def process_is_alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def job_summary(path: Path) -> dict[str, Any]:
  metadata = read_json(path / "job.json")
  run_id = run_id_from_dir(path)
  if run_id is None:
    raise ValueError(f"Cannot determine run_id for {path}")
  status = metadata.get("status", (path / "status").read_text().strip() if (path / "status").exists() else "unknown")
  pid = None
  if (path / "pid").exists():
    try:
      pid = int((path / "pid").read_text().strip())
    except ValueError:
      pid = None
  if status == "running" and pid is not None and not process_is_alive(pid):
    status = "stale"
  return {
    "run_id": run_id,
    "status": status,
    "pid": pid,
    "updated_at": metadata.get("updated_at"),
    "path": str(path),
  }


def list_jobs(job_root: Path) -> list[dict[str, Any]]:
  return sorted((job_summary(path) for path in job_dirs(job_root)), key=lambda row: row["run_id"])
