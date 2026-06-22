from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
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
DEFAULT_UV_EXTRAS = ("plots",)
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
INCLUDED_IGNORED_PATHS = {
  "wandb/config.json",
  "wandb/.gitignore",
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
  uv_extras: tuple[str, ...] = DEFAULT_UV_EXTRAS
  uv_extras_explicit: bool = False
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
    return relative.as_posix() in INCLUDED_IGNORED_PATHS

  if any(part in EXCLUDED_DIRS for part in parts):
    return False

  name = path.name
  return not any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def progress_paths(paths: list[Path], remote: RemoteConfig, desc: str):
  if remote.verbosity == 0 or remote.quiet or tqdm is None:
    return paths
  return tqdm(paths, desc=desc, unit="file")


def file_digest(path: Path) -> tuple[str, int]:
  data = path.read_bytes()
  return hashlib.sha256(data).hexdigest(), len(data)


def uv_lock_digest() -> str:
  return file_digest(REPO_ROOT / "uv.lock")[0]


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
  relative_paths.update(
    relative_path
    for relative_path in INCLUDED_IGNORED_PATHS
    if (REPO_ROOT / relative_path).exists()
  )

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


def ssh_control_command(remote: RemoteConfig, operation: str) -> list[str]:
  cmd = ["ssh", "-S", remote.control_path, "-O", operation, remote.host]
  if remote.quiet:
    cmd.insert(1, "-q")
  elif remote.external_verbosity > 0:
    cmd.insert(1, "-" + "v" * remote.external_verbosity)
  return cmd


def ssh_master_running(remote: RemoteConfig) -> bool:
  cmd = ssh_control_command(remote, "check")
  if remote.show_commands and not remote.quiet:
    print("+", " ".join(shlex.quote(part) for part in cmd))
  if remote.dry_run:
    return False
  result = subprocess.run(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
  )
  return result.returncode == 0


def open_master(remote: RemoteConfig) -> bool:
  if ssh_master_running(remote):
    if remote.verbosity > 0 and not remote.quiet:
      print("reusing existing ssh master")
    return False
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
  return True


def close_master(remote: RemoteConfig) -> None:
  run(
    ssh_control_command(remote, "exit"),
    dry_run=remote.dry_run,
    quiet=remote.quiet,
    show_command=remote.show_commands,
  )


@contextmanager
def ssh_master(remote: RemoteConfig):
  opened_master = open_master(remote)
  try:
    yield
  finally:
    if opened_master and False:
      close_master(remote)


def remote_git_dir(remote: RemoteConfig) -> str:
  return f"{remote.quoted_remote_dir}/.remote/git"


def remote_meta_dir(remote: RemoteConfig) -> str:
  return f"{remote.quoted_remote_dir}/.remote"


def remote_uv_json_path(remote: RemoteConfig) -> str:
  return f"{remote_meta_dir(remote)}/uv.json"


def read_remote_uv_json(remote: RemoteConfig) -> dict:
  raw = ssh_capture(remote, f"cat {remote_uv_json_path(remote)} 2>/dev/null || true")
  if not raw:
    return {}
  try:
    metadata = json.loads(raw)
  except json.JSONDecodeError:
    if remote.verbosity > 0 and not remote.quiet:
      print("ignoring unreadable remote uv metadata")
    return {}
  return metadata if isinstance(metadata, dict) else {}


def metadata_uv_extras(metadata: dict) -> tuple[str, ...] | None:
  extras = metadata.get("uv_extras")
  if not isinstance(extras, list) or not all(isinstance(extra, str) for extra in extras):
    return None
  return tuple(extras)


def resolve_setup_extras(remote: RemoteConfig, setup: SetupConfig, metadata: dict) -> SetupConfig:
  if setup.uv_extras_explicit:
    return setup
  remote_extras = metadata_uv_extras(metadata)
  if remote_extras is None:
    return setup
  if remote_extras != setup.uv_extras and remote.verbosity > 0 and not remote.quiet:
    print(f"using remote uv extras: {', '.join(remote_extras) or '(none)'}")
  return SetupConfig(
    model=setup.model,
    uv_extras=remote_extras,
    uv_extras_explicit=setup.uv_extras_explicit,
    proxy=setup.proxy,
  )


def remote_uv_is_synced(metadata: dict, lock_digest: str, extras: tuple[str, ...]) -> bool:
  return (
    metadata.get("uv_lock_sha256") == lock_digest
    and metadata_uv_extras(metadata) == extras
  )


def write_remote_uv_json_command(lock_digest: str, extras: tuple[str, ...]) -> str:
  payload = json.dumps(
    {
      "uv_lock_sha256": lock_digest,
      "uv_extras": list(extras),
    },
    sort_keys=True,
  )
  script = "\n".join(
    [
      "import datetime",
      "import json",
      "import pathlib",
      "",
      f'metadata = json.loads("""{payload}""")',
      'metadata["synced_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()',
      'path = pathlib.Path(".remote/uv.json")',
      "path.parent.mkdir(parents=True, exist_ok=True)",
      'print(json.dumps(metadata, indent=2, sort_keys=True), file=path.open("w"))',
    ]
  )
  return "\n".join(["{ python - <<PY", script, "PY", "}"])


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


def setup_remote(remote: RemoteConfig, setup: SetupConfig) -> SetupConfig:
  metadata = read_remote_uv_json(remote)
  setup = resolve_setup_extras(remote, setup, metadata)
  lock_digest = uv_lock_digest()
  commands = []
  if remote_uv_is_synced(metadata, lock_digest, setup.uv_extras):
    if remote.verbosity > 0 and not remote.quiet:
      print("remote uv sync is current")
  else:
    commands.append(
      f"{proxy_env(setup)}{uv_sync_prefix(remote, setup)} && "
      f"{write_remote_uv_json_command(lock_digest, setup.uv_extras)}"
    )
  if setup.model == "olmo2-1B":
    commands.append(
      f"( test -d {OLMO2_1B_CACHE_PATH} || "
      f"{proxy_env(setup)}{uv_run_prefix(remote, setup)} hf download "
      f"{shlex.quote(OLMO2_1B_MODEL_ID)} --revision {OLMO2_1B_REVISION} )"
    )
  if commands:
    ssh(remote, f"cd {remote.quoted_remote_dir} && {' && '.join(commands)}")
  return setup


def ensure_remote_ready(remote: RemoteConfig, setup: SetupConfig) -> SetupConfig:
  sync_code(remote)
  return setup_remote(remote, setup)


def remote_train_command(remote: RemoteConfig, setup: SetupConfig, train_args: list[str], log_path: str | None = None) -> str:
  quoted_train_args = " ".join(shlex.quote(arg) for arg in ["--model", setup.model, *train_args])
  train_command = f"{proxy_env(setup)}{uv_run_prefix(remote, setup)} python -u scripts/train.py {quoted_train_args} "
  command = (
    f"cd {remote.quoted_remote_dir} && "
    f"{train_command}"
  )
  if log_path is None:
    return command
  quoted_log_path = remote_quote(log_path)
  logged_command = shell_join([
    "bash",
    "-o",
    "pipefail",
    "-lc",
    f"{train_command}2>&1 | tee {quoted_log_path}",
  ])
  return (
    f"cd {remote.quoted_remote_dir} && "
    f"mkdir -p {remote_quote(str(Path(log_path).parent))} && "
    f"{logged_command}"
  )


def remainder_args(values: list[str]) -> list[str]:
  if values and values[0] == "--":
    return values[1:]
  return values


def without_job_args(values: tuple[str, ...]) -> list[str]:
  args = remainder_args(list(values))
  result = []
  skip_next = False
  options_with_value = {"--job-root", "--run-id", "--checkpoint-every"}
  for arg in args:
    if skip_next:
      skip_next = False
      continue
    if arg in {"--job", "--no-job", "--resume"}:
      continue
    if arg in options_with_value:
      skip_next = True
      continue
    if any(arg.startswith(f"{option}=") for option in options_with_value):
      continue
    result.append(arg)
  return result


def force_job_args(values: tuple[str, ...]) -> list[str]:
  args = remainder_args(list(values))
  return ["--job", *(arg for arg in args if arg not in {"--job", "--no-job"})]


def smoke_train_args(train_args: tuple[str, ...], smoke_group_size: int) -> list[str]:
  return [
    *without_job_args(train_args),
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
  uv_extras = getattr(args, "uv_extras", None)
  return SetupConfig(
    model=getattr(args, "model", "tiny"),
    uv_extras=tuple(uv_extras or DEFAULT_UV_EXTRAS),
    uv_extras_explicit=uv_extras is not None,
    proxy=getattr(args, "proxy", None),
  )


def local_host_dir_name(host: str) -> str:
  hostname = host.rsplit("@", 1)[-1]
  safe_hostname = "".join(char if char.isalnum() or char in ".-" else "_" for char in hostname)
  return safe_hostname.strip("._") or "remote"


def download_results(remote: RemoteConfig, local_out: str) -> None:
  host_out = Path(local_out) / local_host_dir_name(remote.host)
  local_jobs = host_out / "jobs"
  local_wandb = host_out / "wandb"
  local_jobs.mkdir(parents=True, exist_ok=True)
  local_wandb.mkdir(parents=True, exist_ok=True)
  run([
    "rsync",
    "-az",
    "-e",
    rsync_ssh(remote),
    f"{remote.host}:{remote_quote(remote.remote_dir.rstrip('/') + '/out/jobs/')}",
    f"{local_jobs}/",
  ], dry_run=remote.dry_run, quiet=remote.quiet, show_command=remote.show_commands)
  run([
    "rsync",
    "-az",
    "-e",
    rsync_ssh(remote),
    f"{remote.host}:{remote_quote(remote.remote_dir.rstrip('/') + '/wandb/')}",
    f"{local_wandb}/",
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
    setup = ensure_remote_ready(remote, setup)
    run_smoke(remote, setup, train_args, args.smoke_group_size, args.smoke_log_path)


def cmd_train(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  setup = setup_from_args(args)
  train_args = tuple(remainder_args(args.train_args))
  with ssh_master(remote):
    setup = ensure_remote_ready(remote, setup)
    run_smoke(remote, setup, train_args, args.smoke_group_size, args.smoke_log_path)
    ssh(remote, remote_train_command(remote, setup, force_job_args(train_args)))


def cmd_download(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  with ssh_master(remote):
    download_results(remote, args.local_out)


def remote_jobs_command(remote: RemoteConfig, setup: SetupConfig, job_args: tuple[str, ...]) -> str:
  quoted_job_args = " ".join(shlex.quote(arg) for arg in remainder_args(list(job_args)))
  command = f"{proxy_env(setup)}{uv_run_prefix(remote, setup)} python scripts/jobs.py"
  if quoted_job_args:
    command = f"{command} {quoted_job_args}"
  return f"cd {remote.quoted_remote_dir} && {command}"


def cmd_jobs(args: argparse.Namespace) -> None:
  remote = remote_from_args(args)
  setup = setup_from_args(args)
  with ssh_master(remote):
    setup = ensure_remote_ready(remote, setup)
    ssh(remote, remote_jobs_command(remote, setup, tuple(args.job_args)))


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
  smoke_parser.set_defaults(train_args=[])
  smoke_parser.set_defaults(func=cmd_smoke)

  train_parser = subparsers.add_parser("train")
  add_setup_args(train_parser)
  train_parser.add_argument("--smoke-group-size", type=int, default=2)
  train_parser.add_argument("--smoke-log-path", default="out/smoke.log")
  train_parser.set_defaults(train_args=[])
  train_parser.set_defaults(func=cmd_train)

  download_parser = subparsers.add_parser("download")
  download_parser.add_argument("--local-out", default="out")
  download_parser.set_defaults(func=cmd_download)

  jobs_parser = subparsers.add_parser("jobs")
  add_setup_args(jobs_parser)
  jobs_parser.set_defaults(job_args=[])
  jobs_parser.set_defaults(func=cmd_jobs)

  args, passthrough_args = parser.parse_known_args()
  if passthrough_args:
    if args.command in {"smoke", "train"}:
      args.train_args = [*args.train_args, *passthrough_args]
    elif args.command == "jobs":
      args.job_args = [*args.job_args, *passthrough_args]
    else:
      parser.error(f"unrecognized arguments: {' '.join(passthrough_args)}")
  args.func(args)


if __name__ == "__main__":
  main()
