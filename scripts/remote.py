from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shlex
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
  from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for this helper.
  tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL_PATH = "~/.ssh/cm-%r@%h:%p"
OLMO2_1B_MODEL_ID = "allenai/OLMo-2-0425-1B"
OLMO2_1B_REVISION = "a1847dff35000b4271fa70afc5db10fd29fedbdf"
OLMO2_1B_CACHE_PATH = (
  "$HOME/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/"
  f"snapshots/{OLMO2_1B_REVISION}"
)

EXCLUDED_DIRS = {
  ".git",
  ".venv",
  "__pycache__",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
  ".pyre",
  ".hypothesis",
  "alignment.egg-info",
  "out",
}

EXCLUDED_FILE_PATTERNS = {
  ".env",
  "*.pyc",
  "*.pyo",
  "*.log",
  "*.tmp",
  ".DS_Store",
}


@dataclass(frozen=True)
class RemoteConfig:
  host: str
  remote_dir: str
  control_path: str
  control_persist: str
  dry_run: bool = False
  verbosity: int = 0
  quiet: bool = False

  @property
  def quoted_remote_dir(self) -> str:
    return remote_quote(self.remote_dir)

  @property
  def external_verbosity(self) -> int:
    return max(0, min(self.verbosity - 1, 3))

  @property
  def show_commands(self) -> bool:
    return self.verbosity > 0 or self.dry_run


@dataclass(frozen=True)
class SetupConfig:
  model: str = "tiny"
  uv_extras: tuple[str, ...] = ("plots",)
  proxy: str | None = None


def run(
  cmd: list[str],
  dry_run: bool = False,
  quiet: bool = False,
  show_command: bool = False,
) -> subprocess.CompletedProcess:
  if show_command and not quiet:
    print("+", " ".join(shlex.quote(part) for part in cmd))
  if dry_run:
    return subprocess.CompletedProcess(cmd, 0, "", "")
  result = subprocess.run(cmd, check=False, text=True)
  if result.returncode != 0:
    raise SystemExit(result.returncode)
  return result


def capture(cmd: list[str], dry_run: bool = False, quiet: bool = False, show_command: bool = False) -> str:
  if show_command and not quiet:
    print("+", " ".join(shlex.quote(part) for part in cmd))
  if dry_run:
    return ""
  result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, text=True)
  if result.returncode != 0:
    raise SystemExit(result.returncode)
  return result.stdout.strip()


def ssh_base(remote: RemoteConfig) -> list[str]:
  cmd = [
    "ssh",
    "-S",
    remote.control_path,
    "-o",
    "ControlMaster=auto",
    "-o",
    f"ControlPersist={remote.control_persist}",
  ]
  if remote.quiet:
    cmd.append("-q")
  elif remote.external_verbosity > 0:
    cmd.append("-" + "v" * remote.external_verbosity)
  return cmd


def ssh(remote: RemoteConfig, remote_command: str) -> None:
  run(
    [*ssh_base(remote), remote.host, remote_command],
    dry_run=remote.dry_run,
    quiet=remote.quiet,
    show_command=remote.show_commands,
  )


def ssh_capture(remote: RemoteConfig, remote_command: str) -> str:
  return capture(
    [*ssh_base(remote), remote.host, remote_command],
    dry_run=remote.dry_run,
    quiet=remote.quiet,
    show_command=remote.show_commands,
  )


def rsync_ssh(remote: RemoteConfig) -> str:
  return " ".join(shlex.quote(part) for part in ssh_base(remote))


def should_include(path: Path) -> bool:
  relative = path.relative_to(REPO_ROOT)
  parts = relative.parts

  if parts and parts[0] == "wandb":
    return relative.as_posix() in {"wandb/config.json", "wandb/.gitignore"}

  if any(part in EXCLUDED_DIRS for part in parts):
    return False

  name = path.name
  return not any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def progress_paths(paths: list[Path], remote: RemoteConfig, desc: str):
  if remote.quiet or tqdm is None:
    return paths
  return tqdm(paths, desc=desc, unit="file")


def file_digest(path: Path) -> tuple[str, int]:
  data = path.read_bytes()
  return hashlib.sha256(data).hexdigest(), len(data)


def git_output(args: list[str]) -> str:
  return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def git_head() -> str:
  return git_output(["rev-parse", "HEAD"])


def local_has_commit(commit: str) -> bool:
  return subprocess.run(
    ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
    cwd=REPO_ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
  ).returncode == 0


def build_source_bundle(remote: RemoteConfig, base_commit: str | None) -> tuple[Path, str, tempfile.TemporaryDirectory]:
  tmpdir = tempfile.TemporaryDirectory(prefix="assignment5-remote-")
  bundle_path = Path(tmpdir.name) / "source.bundle"
  refspec = [f"{base_commit}..HEAD"] if base_commit else ["HEAD"]
  if remote.verbosity > 0 and not remote.quiet:
    print(f"source bundle refspec: {' '.join(refspec)}")
  run(
    ["git", "bundle", "create", str(bundle_path), *refspec],
    quiet=remote.quiet,
    show_command=remote.show_commands,
  )
  digest, size = file_digest(bundle_path)
  print(f"source bundle sha256={digest} size={size} bytes")
  return bundle_path, digest, tmpdir


def changed_source_paths() -> tuple[list[Path], list[str]]:
  changed_output = subprocess.check_output(
    ["git", "diff", "--name-only", "-z", "HEAD", "--"],
    cwd=REPO_ROOT,
  )
  untracked_output = subprocess.check_output(
    ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    cwd=REPO_ROOT,
  )
  relative_paths = {
    path
    for path in changed_output.decode().split("\0") + untracked_output.decode().split("\0")
    if path
  }

  files: list[Path] = []
  deleted: list[str] = []
  for relative_path in sorted(relative_paths):
    path = REPO_ROOT / relative_path
    if not should_include(path):
      continue
    if path.is_file():
      files.append(path)
    else:
      deleted.append(relative_path)
  return files, deleted


def build_patch_zip(remote: RemoteConfig) -> tuple[Path, str, tempfile.TemporaryDirectory]:
  tmpdir = tempfile.TemporaryDirectory(prefix="assignment5-patch-")
  zip_path = Path(tmpdir.name) / "patch.zip"
  files, deleted = changed_source_paths()

  if remote.verbosity > 0 and not remote.quiet:
    print(f"patch zip file count: {len(files)} deleted={len(deleted)}")
  if remote.verbosity > 1 and not remote.quiet:
    for path in files:
      print(path.relative_to(REPO_ROOT).as_posix())
    for path in deleted:
      print(f"{path} [deleted]")

  with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(".deleted", "\n".join(deleted) + ("\n" if deleted else ""))
    for path in progress_paths(files, remote, "Building patch zip"):
      archive.write(path, path.relative_to(REPO_ROOT))

  digest, size = file_digest(zip_path)
  print(f"patch zip sha256={digest} size={size} bytes")
  return zip_path, digest, tmpdir


def remote_quote(path: str | Path) -> str:
  return shlex.quote(str(path))


def shell_join(parts: list[str]) -> str:
  return " ".join(shlex.quote(part) for part in parts)


def proxy_env(setup: SetupConfig) -> str:
  if setup.proxy is None:
    return ""
  proxy = shlex.quote(setup.proxy)
  return f"HTTP_PROXY={proxy} HTTPS_PROXY={proxy} ALL_PROXY={proxy} "


def uv_extra_args(setup: SetupConfig) -> list[str]:
  args = []
  for extra in setup.uv_extras:
    args.extend(["--extra", extra])
  return args


def verbosity_args(remote: RemoteConfig) -> list[str]:
  if remote.quiet:
    return ["-q"]
  if remote.external_verbosity > 0:
    return ["-" + "v" * remote.external_verbosity]
  return []


def uv_sync_prefix(remote: RemoteConfig, setup: SetupConfig) -> str:
  return shell_join(["uv", *verbosity_args(remote), "sync", *uv_extra_args(setup)])


def uv_run_prefix(remote: RemoteConfig, setup: SetupConfig) -> str:
  return shell_join(["uv", *verbosity_args(remote), "run", "--offline", *uv_extra_args(setup)])


def open_master(remote: RemoteConfig) -> None:
  cmd = [
    "ssh",
    "-M",
    "-S",
    remote.control_path,
    "-o",
    f"ControlPersist={remote.control_persist}",
    "-fN",
    remote.host,
  ]
  if remote.quiet:
    cmd.insert(1, "-q")
  elif remote.external_verbosity > 0:
    cmd.insert(1, "-" + "v" * remote.external_verbosity)
  run(cmd, dry_run=remote.dry_run, quiet=remote.quiet, show_command=remote.show_commands)


def close_master(remote: RemoteConfig) -> None:
  cmd = ["ssh", "-S", remote.control_path, "-O", "exit", remote.host]
  if remote.quiet:
    cmd.insert(1, "-q")
  elif remote.external_verbosity > 0:
    cmd.insert(1, "-" + "v" * remote.external_verbosity)
  run(cmd, dry_run=remote.dry_run, quiet=remote.quiet, show_command=remote.show_commands)


@contextmanager
def ssh_master(remote: RemoteConfig):
  open_master(remote)
  try:
    yield
  finally:
    close_master(remote)


def remote_git_dir(remote: RemoteConfig) -> str:
  return f"{remote.quoted_remote_dir}/.remote/git"


def remote_meta_dir(remote: RemoteConfig) -> str:
  return f"{remote.quoted_remote_dir}/.remote"


def remote_has_commit(remote: RemoteConfig, commit: str) -> bool:
  result = ssh_capture(
    remote,
    f"git --git-dir {remote_git_dir(remote)} cat-file -e {shlex.quote(commit)}^{{commit}} 2>/dev/null && echo yes || true",
  )
  return result == "yes"


def remote_synced_head(remote: RemoteConfig) -> str:
  return ssh_capture(
    remote,
    f"git --git-dir {remote_git_dir(remote)} rev-parse refs/heads/synced 2>/dev/null || true",
  )


def upload_file(remote: RemoteConfig, local_path: Path, remote_path: str) -> None:
  run(
    [
      "rsync",
      "-az",
      "-e",
      rsync_ssh(remote),
      str(local_path),
      f"{remote.host}:{remote_path}",
    ],
    dry_run=remote.dry_run,
    quiet=remote.quiet,
    show_command=remote.show_commands,
  )


def sync_bundle(remote: RemoteConfig, head: str) -> None:
  meta = remote_meta_dir(remote)
  git_dir = remote_git_dir(remote)
  ssh(remote, f"mkdir -p {meta}")
  ssh(remote, f"test -d {git_dir} || git init --bare {git_dir}")

  if remote_has_commit(remote, head):
    if remote.verbosity > 0 and not remote.quiet:
      print(f"remote already has commit: {head}")
    ssh(remote, f"git --git-dir {git_dir} update-ref refs/heads/synced {shlex.quote(head)}")
    return

  base_commit = remote_synced_head(remote)
  if not base_commit or not local_has_commit(base_commit):
    base_commit = None

  bundle_path, bundle_digest, tmpdir = build_source_bundle(remote, base_commit)
  try:
    upload_file(remote, bundle_path, f"{meta}/source.bundle")
    ssh(
      remote,
      (
        f"cd {remote.quoted_remote_dir} && "
        f"git --git-dir .remote/git fetch .remote/source.bundle +HEAD:refs/heads/synced && "
        f"printf %s {shlex.quote(bundle_digest)} > .remote/source.bundle.sha256"
      ),
    )
  finally:
    tmpdir.cleanup()


def checkout_synced_commit(remote: RemoteConfig, head: str) -> None:
  ssh(
    remote,
    (
      f"cd {remote.quoted_remote_dir} && "
      f"git --git-dir .remote/git --work-tree . checkout -f {shlex.quote(head)}"
    ),
  )


def remote_patch_apply_script(patch_digest: str) -> str:
  return "\n".join(
    [
      "import pathlib, shutil, zipfile",
      'patch_dir = pathlib.Path(".remote/patch")',
      "shutil.rmtree(patch_dir, ignore_errors=True)",
      "patch_dir.mkdir(parents=True, exist_ok=True)",
      'with zipfile.ZipFile(".remote/patch.zip") as archive:',
      "  archive.extractall(patch_dir)",
      'deleted = patch_dir / ".deleted"',
      "if deleted.exists():",
      "  for line in deleted.read_text().splitlines():",
      "    if line:",
      "      path = pathlib.Path(line)",
      "      if path.exists() or path.is_symlink():",
      "        path.unlink()",
      "  deleted.unlink()",
      'for src in patch_dir.rglob("*"):',
      "  if src.is_dir():",
      "    continue",
      "  dst = pathlib.Path(src.relative_to(patch_dir))",
      "  dst.parent.mkdir(parents=True, exist_ok=True)",
      "  shutil.copy2(src, dst)",
      f'pathlib.Path(".remote/patch.sha256").write_text("{patch_digest}")',
    ]
  )


def apply_patch_zip(remote: RemoteConfig) -> None:
  patch_path, patch_digest, tmpdir = build_patch_zip(remote)
  try:
    meta = remote_meta_dir(remote)
    upload_file(remote, patch_path, f"{meta}/patch.zip")
    script = remote_patch_apply_script(patch_digest)
    ssh(
      remote,
      (
        f"cd {remote.quoted_remote_dir} && "
        "python3 - <<PY\n"
        f"{script}\n"
        "PY"
      ),
    )
  finally:
    tmpdir.cleanup()


def sync_code(remote: RemoteConfig) -> None:
  head = git_head()
  if remote.verbosity > 0 and not remote.quiet:
    print(f"local HEAD: {head}")
  sync_bundle(remote, head)
  checkout_synced_commit(remote, head)
  apply_patch_zip(remote)


def setup_remote(remote: RemoteConfig, setup: SetupConfig) -> None:
  commands = [f"{proxy_env(setup)}{uv_sync_prefix(remote, setup)}"]
  if setup.model == "olmo2-1B":
    commands.append(
      f"( test -d {OLMO2_1B_CACHE_PATH} || "
      f"{proxy_env(setup)}{uv_run_prefix(remote, setup)} hf download "
      f"{shlex.quote(OLMO2_1B_MODEL_ID)} --revision {OLMO2_1B_REVISION} )"
    )
  ssh(remote, f"cd {remote.quoted_remote_dir} && {' && '.join(commands)}")


def ensure_remote_ready(remote: RemoteConfig, setup: SetupConfig) -> None:
  sync_code(remote)
  setup_remote(remote, setup)


def remote_train_command(remote: RemoteConfig, setup: SetupConfig, train_args: list[str], log_path: str) -> str:
  quoted_train_args = " ".join(shlex.quote(arg) for arg in ["--model", setup.model, *train_args])
  quoted_log_path = remote_quote(log_path)
  return (
    f"cd {remote.quoted_remote_dir} && "
    f"mkdir -p {remote_quote(str(Path(log_path).parent))} && "
    f"{proxy_env(setup)}{uv_run_prefix(remote, setup)} python -u scripts/train.py {quoted_train_args} "
    f"2>&1 | tee {quoted_log_path}"
  )


def remainder_args(values: list[str]) -> list[str]:
  if values and values[0] == "--":
    return values[1:]
  return values


def smoke_train_args(train_args: tuple[str, ...], smoke_group_size: int) -> list[str]:
  return [
    *remainder_args(list(train_args)),
    "--wandb-mode",
    "disabled",
    "--group-size",
    str(smoke_group_size),
    "--rollout-batch-size",
    str(smoke_group_size),
    "--num-rollout-steps",
    "3",
  ]


def run_smoke(
  remote: RemoteConfig,
  setup: SetupConfig,
  train_args: tuple[str, ...],
  smoke_group_size: int,
  smoke_log_path: str,
) -> None:
  ssh(
    remote,
    remote_train_command(remote, setup, smoke_train_args(train_args, smoke_group_size), smoke_log_path),
  )


def remote_from_args(args: argparse.Namespace) -> RemoteConfig:
  return RemoteConfig(
    host=args.host,
    remote_dir=args.remote_dir,
    control_path=args.control_path,
    control_persist=args.control_persist,
    dry_run=args.dry_run,
    verbosity=args.verbosity,
    quiet=args.quiet,
  )


def setup_from_args(args: argparse.Namespace) -> SetupConfig:
  return SetupConfig(
    model=getattr(args, "model", "tiny"),
    uv_extras=tuple(getattr(args, "uv_extras", None) or ("plots",)),
    proxy=getattr(args, "proxy", None),
  )


def download_results(remote: RemoteConfig, local_out: str, local_wandb: str) -> None:
  Path(local_out).mkdir(parents=True, exist_ok=True)
  Path(local_wandb).mkdir(parents=True, exist_ok=True)
  run([
    "rsync",
    "-az",
    "-e",
    rsync_ssh(remote),
    f"{remote.host}:{remote_quote(remote.remote_dir.rstrip('/') + '/out/')}",
    f"{local_out.rstrip('/')}/",
  ], dry_run=remote.dry_run, quiet=remote.quiet, show_command=remote.show_commands)
  run([
    "rsync",
    "-az",
    "-e",
    rsync_ssh(remote),
    f"{remote.host}:{remote_quote(remote.remote_dir.rstrip('/') + '/wandb/')}",
    f"{local_wandb.rstrip('/')}/",
  ], dry_run=remote.dry_run, quiet=remote.quiet, show_command=remote.show_commands)


def cmd_sync(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  with ssh_master(remote):
    sync_code(remote)


def cmd_setup(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  setup = setup_from_args(args)
  with ssh_master(remote):
    setup_remote(remote, setup)


def cmd_smoke(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  setup = setup_from_args(args)
  train_args = tuple(remainder_args(args.train_args))
  with ssh_master(remote):
    ensure_remote_ready(remote, setup)
    run_smoke(remote, setup, train_args, args.smoke_group_size, args.smoke_log_path)


def cmd_train(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  setup = setup_from_args(args)
  train_args = tuple(remainder_args(args.train_args))
  with ssh_master(remote):
    ensure_remote_ready(remote, setup)
    run_smoke(remote, setup, train_args, args.smoke_group_size, args.smoke_log_path)
    ssh(remote, remote_train_command(remote, setup, list(train_args), args.log_path))


def cmd_download(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  with ssh_master(remote):
    download_results(remote, args.local_out, args.local_wandb)


def add_common_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--host", required=True)
  parser.add_argument("--remote-dir", required=True)
  parser.add_argument("--control-path", default=os.path.expanduser(DEFAULT_CONTROL_PATH))
  parser.add_argument("--control-persist", default="30m")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("-v", dest="verbosity", action="count", default=0)
  parser.add_argument("-q", "--quiet", action="store_true")


def add_setup_args(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--model", choices=["tiny", "olmo2-1B"], default="tiny")
  parser.add_argument("--uv-extra", dest="uv_extras", action="append")
  parser.add_argument("--proxy")


def main() -> None:
  parser = argparse.ArgumentParser(description="Remote helper for assignment5 GRPO runs.")
  add_common_args(parser)
  subparsers = parser.add_subparsers(dest="command", required=True)

  sync_parser = subparsers.add_parser("sync")
  sync_parser.set_defaults(func=cmd_sync)

  setup_parser = subparsers.add_parser("setup")
  add_setup_args(setup_parser)
  setup_parser.set_defaults(func=cmd_setup)

  smoke_parser = subparsers.add_parser("smoke")
  add_setup_args(smoke_parser)
  smoke_parser.add_argument("--smoke-group-size", type=int, default=2)
  smoke_parser.add_argument("--smoke-log-path", default="out/smoke.log")
  smoke_parser.add_argument("train_args", nargs=argparse.REMAINDER)
  smoke_parser.set_defaults(func=cmd_smoke)

  train_parser = subparsers.add_parser("train")
  add_setup_args(train_parser)
  train_parser.add_argument("--log-path", default="out/train.log")
  train_parser.add_argument("--smoke-group-size", type=int, default=2)
  train_parser.add_argument("--smoke-log-path", default="out/smoke.log")
  train_parser.add_argument("train_args", nargs=argparse.REMAINDER)
  train_parser.set_defaults(func=cmd_train)

  download_parser = subparsers.add_parser("download")
  download_parser.add_argument("--local-out", default="out")
  download_parser.add_argument("--local-wandb", default="wandb")
  download_parser.set_defaults(func=cmd_download)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
