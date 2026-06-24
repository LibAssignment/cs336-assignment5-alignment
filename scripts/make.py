from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train.py"
JOBS_SCRIPT = REPO_ROOT / "scripts" / "jobs.py"

JOB_FLAGS = {"--job", "--no-job"}
SMOKE_OPTIONS_WITH_VALUE = {
  "--job-root",
  "--run-id",
  "--checkpoint-every",
  "--rollout-every",
  "--validate-every",
}


@dataclass(frozen=True)
class RunOptions:
  dry_run: bool = False
  verbosity: int = 0
  quiet: bool = False

  @property
  def show_command(self) -> bool:
    return self.dry_run or self.verbosity > 0


def shell_join(argv: list[str]) -> str:
  return " ".join(shlex.quote(arg) for arg in argv)


def run(argv: list[str], options: RunOptions) -> None:
  if options.show_command and not options.quiet:
    print("+", shell_join(argv))
  if options.dry_run:
    return
  result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
  if result.returncode != 0:
    raise SystemExit(result.returncode)


def passthrough_args(values: list[str]) -> list[str]:
  if values and values[0] == "--":
    return values[1:]
  return values


def without_options_with_values(values: list[str], options: set[str]) -> list[str]:
  result = []
  skip_next = False
  for value in values:
    if skip_next:
      skip_next = False
      continue
    if value in options:
      skip_next = True
      continue
    if any(value.startswith(f"{option}=") for option in options):
      continue
    result.append(value)
  return result


def extract_job_mode(values: list[str], default: str | None) -> tuple[str | None, list[str]]:
  mode = default
  result = []
  for value in values:
    if value == "--job":
      if mode == "no-job":
        raise SystemExit("--job cannot be combined with --no-job")
      mode = "job"
      continue
    if value == "--no-job":
      if mode == "job":
        raise SystemExit("--job cannot be combined with --no-job")
      mode = "no-job"
      continue
    result.append(value)
  return mode, result


def train_command(train_args: list[str]) -> list[str]:
  return [sys.executable, str(TRAIN_SCRIPT), *train_args]


def jobs_command(job_args: list[str]) -> list[str]:
  return [sys.executable, str(JOBS_SCRIPT), *job_args]


def smoke_train_args(train_args: list[str], smoke_group_size: int) -> list[str]:
  args = without_options_with_values(train_args, SMOKE_OPTIONS_WITH_VALUE)
  args = [arg for arg in args if arg not in JOB_FLAGS and arg != "--resume"]
  return [
    *args,
    "--no-job",
    "--wandb-mode",
    "disabled",
    "--group-size",
    str(smoke_group_size),
    "--rollout-batch-size",
    str(smoke_group_size),
    "--num-rollout-steps",
    "3",
  ]


def train_args_with_job_mode(train_args: list[str], job_mode: str | None) -> list[str]:
  if job_mode == "job":
    return ["--job", *train_args]
  if job_mode == "no-job":
    return ["--no-job", *train_args]
  return train_args


def cmd_smoke(args: argparse.Namespace) -> None:
  train_args = passthrough_args(args.train_args)
  options = RunOptions(dry_run=args.dry_run, verbosity=args.verbosity, quiet=args.quiet)
  run(train_command(smoke_train_args(train_args, args.smoke_group_size)), options)


def cmd_train(args: argparse.Namespace) -> None:
  train_args = passthrough_args(args.train_args)
  job_mode, train_args = extract_job_mode(train_args, args.job_mode)
  options = RunOptions(dry_run=args.dry_run, verbosity=args.verbosity, quiet=args.quiet)
  if args.smoke:
    run(train_command(smoke_train_args(train_args, args.smoke_group_size)), options)
  run(train_command(train_args_with_job_mode(train_args, job_mode)), options)


def cmd_jobs(args: argparse.Namespace) -> None:
  job_args = passthrough_args(args.job_args)
  options = RunOptions(dry_run=args.dry_run, verbosity=args.verbosity, quiet=args.quiet)
  run(jobs_command(job_args), options)


def add_common_args(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
  default = argparse.SUPPRESS if suppress_defaults else None
  parser.add_argument("--dry-run", action="store_true", default=default)
  parser.add_argument("-v", dest="verbosity", action="count", default=argparse.SUPPRESS if suppress_defaults else 0)
  parser.add_argument("-q", "--quiet", action="store_true", default=default)


def add_train_passthrough(parser: argparse.ArgumentParser) -> None:
  parser.set_defaults(train_args=[])


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Local assignment5 training helper. Remote sync/setup/run lives in expri."
  )
  add_common_args(parser)
  subparsers = parser.add_subparsers(dest="command", required=True)

  smoke_parser = subparsers.add_parser("smoke", help="run a short local smoke training pass")
  add_common_args(smoke_parser, suppress_defaults=True)
  smoke_parser.add_argument("--smoke-group-size", type=int, default=2)
  add_train_passthrough(smoke_parser)
  smoke_parser.set_defaults(func=cmd_smoke)

  train_parser = subparsers.add_parser("train", help="run local training, optionally after smoke")
  add_common_args(train_parser, suppress_defaults=True)
  smoke_group = train_parser.add_mutually_exclusive_group()
  smoke_group.add_argument("--smoke", dest="smoke", action="store_true", default=True)
  smoke_group.add_argument("--no-smoke", dest="smoke", action="store_false")
  train_parser.add_argument("--smoke-group-size", type=int, default=2)
  job_group = train_parser.add_mutually_exclusive_group()
  job_group.add_argument("--job", dest="job_mode", action="store_const", const="job", default="job")
  job_group.add_argument("--no-job", dest="job_mode", action="store_const", const="no-job")
  add_train_passthrough(train_parser)
  train_parser.set_defaults(func=cmd_train)

  jobs_parser = subparsers.add_parser("jobs", help="run the local jobs inspector")
  add_common_args(jobs_parser, suppress_defaults=True)
  jobs_parser.set_defaults(job_args=[])
  jobs_parser.set_defaults(func=cmd_jobs)

  args, unknown_args = parser.parse_known_args()
  if unknown_args:
    if hasattr(args, "train_args"):
      args.train_args = [*args.train_args, *unknown_args]
    elif hasattr(args, "job_args"):
      args.job_args = [*args.job_args, *unknown_args]
    else:
      parser.error(f"unrecognized arguments: {' '.join(unknown_args)}")
  args.func(args)


if __name__ == "__main__":
  main()
