from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

from cs336_alignment.drgrpo_grader import extract_answer, question_only_reward_fn, r1_zero_reward_fn


RewardFn = Callable[[str, str], dict[str, float]]
ExtractFn = Callable[[str], str | None]


@dataclass(frozen=True)
class Prompt:
  name: str
  template: str
  extract: ExtractFn
  reward_fn: RewardFn

  def format(self, question: str) -> str:
    return self.template.format(question=question)


PROMPT_KINDS = [
  "question_only",
  "r1_zero",
  "r1_zero_three_shot_gsm8k",
]

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

PROMPT_REWARD_FNS: dict[str, RewardFn] = {
  "question_only": question_only_reward_fn,
  "r1_zero": r1_zero_reward_fn,
  "r1_zero_three_shot_gsm8k": r1_zero_reward_fn,
}


def load_prompt_template(name: str) -> str:
  with open(PROMPTS_DIR / f"{name}.prompt") as f:
    return f.read()


def load_prompts() -> list[Prompt]:
  return [
    Prompt(
      name=name,
      template=load_prompt_template(name),
      extract=extract_answer,
      reward_fn=PROMPT_REWARD_FNS[name],
    )
    for name in PROMPT_KINDS
  ]


def get_prompt(name: str) -> Prompt:
  if name not in PROMPT_KINDS:
    raise ValueError(f"Unknown prompt kind: {name}")
  return Prompt(
    name=name,
    template=load_prompt_template(name),
    extract=extract_answer,
    reward_fn=PROMPT_REWARD_FNS[name],
  )


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


def make_prompt_rollouts(
  prompt: Prompt,
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
    wrong_answer = make_wrong_answer(answer)
    if prompt.name == "question_only":
      rollout_outputs = [
        f"  correct. \\boxed{{{answer}}}",
        f"  incorrect. \\boxed{{{wrong_answer}}}",
      ]
    else:
      rollout_outputs = [
        f" correct. </think> <answer> {answer} </answer>",
        f" incorrect. </think> <answer> {wrong_answer} </answer>",
      ]
    rollout_outputs.extend(rollout_outputs[-1] for _ in range(group_size - 2))

    prompts.extend([prompt.format(question=example["question"])] * group_size)
    outputs.extend(rollout_outputs)
    ground_truths.extend([answer] * group_size)

  return prompts, outputs, ground_truths
