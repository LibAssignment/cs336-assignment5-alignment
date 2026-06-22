from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cs336_alignment.rl.config import parse_train_config
from cs336_alignment.rl.train import train


def main() -> None:
  parsed_config = parse_train_config()
  train(parsed_config.train, parsed_config.wandb, parsed_config.job)


if __name__ == "__main__":
  main()
