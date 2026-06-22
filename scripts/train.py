from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from cs336_alignment.rl.grpo import grpo_train_step
from cs336_alignment.rl.models import tiny_train_model


def extract_gsm8k_answer(answer: str) -> str:
  if "####" in answer:
    return answer.rsplit("####", maxsplit=1)[-1].strip().replace(",", "")
  matches = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
  if not matches:
    raise ValueError(f"Could not find numeric answer in: {answer!r}")
  return matches[-1]


def make_wrong_answer(answer: str) -> str:
  try:
    value = int(answer)
    return str(value + 1)
  except ValueError:
    value = float(answer)
    return f"{value + 1:g}"


def load_gsm8k_examples(path: Path, limit: int) -> list[dict[str, str]]:
  examples: list[dict[str, str]] = []
  with path.open() as f:
    for line in f:
      raw = json.loads(line)
      examples.append(
        {
          "question": raw["question"],
          "answer": extract_gsm8k_answer(raw["answer"]),
        }
      )
      if len(examples) == limit:
        break
  return examples


def make_prompt(question: str) -> str:
  question_words = question.split()[:8]
  return "Question: " + " ".join(question_words) + " Answer:"


def make_rollouts(
  examples: list[dict[str, str]],
  group_size: int,
) -> tuple[list[str], list[str], list[str]]:
  if group_size < 2:
    raise ValueError("Use group_size >= 2 so each prompt has a correct and incorrect rollout.")

  prompts: list[str] = []
  outputs: list[str] = []
  ground_truths: list[str] = []
  for example in examples:
    answer = example["answer"]
    rollout_outputs = [f"#### {answer}", f"#### {make_wrong_answer(answer)}"]
    rollout_outputs.extend(f"#### {make_wrong_answer(answer)}" for _ in range(group_size - 2))

    prompt = make_prompt(example["question"])
    prompts.extend([prompt] * group_size)
    outputs.extend(rollout_outputs)
    ground_truths.extend([answer] * group_size)

  return prompts, outputs, ground_truths


def build_tokenizer(texts: list[str]) -> PreTrainedTokenizerFast:
  tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.train_from_iterator(
    texts,
    trainer=WordLevelTrainer(
      special_tokens=["<pad>", "<eos>", "<unk>"],
      min_frequency=1,
    ),
  )
  return PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="<pad>",
    eos_token="<eos>",
    unk_token="<unk>",
  )


def gsm8k_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
  parsed = extract_gsm8k_answer(response)
  answer_reward = float(parsed == ground_truth)
  format_reward = float("####" in response)
  return {
    "reward": answer_reward,
    "format_reward": format_reward,
    "answer_reward": answer_reward,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Local smoke test for grpo_train_step on GSM8K.")
  parser.add_argument("--data-path", type=Path, default=Path("data/gsm8k/train.jsonl"))
  parser.add_argument("--num-prompts", type=int, default=2)
  parser.add_argument("--group-size", type=int, default=2)
  parser.add_argument("--steps", type=int, default=1)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
  parser.add_argument("--max-grad-norm", type=float, default=1.0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  torch.manual_seed(0)

  examples = load_gsm8k_examples(args.data_path, args.num_prompts)
  prompts, outputs, ground_truths = make_rollouts(examples, args.group_size)
  tokenizer = build_tokenizer(prompts + outputs + ground_truths)
  model = tiny_train_model(tokenizer).cpu()
  optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

  for step in range(args.steps):
    result = grpo_train_step(
      model=model,
      tokenizer=tokenizer,
      optimizer=optimizer,
      reward_fn=gsm8k_reward_fn,
      prompt_strs=prompts,
      output_strs=outputs,
      ground_truths=ground_truths,
      group_size=args.group_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      max_grad_norm=args.max_grad_norm,
    )
    print(
      "step={step} loss={loss:.6f} reward_mean={reward_mean:.3f} "
      "advantage_mean={advantage_mean:.3f} log_prob_shape={log_prob_shape}".format(
        step=step,
        loss=result.loss.item(),
        reward_mean=result.rewards.float().mean().item(),
        advantage_mean=result.advantages.float().mean().item(),
        log_prob_shape=tuple(result.log_probs.shape),
      )
    )


if __name__ == "__main__":
  main()
