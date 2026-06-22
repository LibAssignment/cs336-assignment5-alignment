from __future__ import annotations

from cs336_alignment.rl.config import parse_train_config
from cs336_alignment.rl.train import train


def main() -> None:
  parsed_config = parse_train_config()
  train(parsed_config.train, parsed_config.wandb)


if __name__ == "__main__":
  main()
