from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Literal

from cs336_alignment.drgrpo_grader import extract_answer, question_only_reward_fn, r1_zero_reward_fn
from ..vllm_utils import VLLMServer


RewardFn = Callable[[str, str], dict[str, float]]
ExtractFn = Callable[[str], str | None]
RewardMode = Literal["answer", "answer+format"]
RewardFormatMode = Literal["loose", "strict"]


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


def with_reward_mode(reward_fn: RewardFn, reward_mode: RewardMode) -> RewardFn:
  if reward_mode == "answer":
    return reward_fn
  if reward_mode != "answer+format":
    raise ValueError(f"Unknown reward mode: {reward_mode}")

  def answer_plus_format_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    rewards = reward_fn(response, ground_truth)
    return {
      **rewards,
      "reward": rewards["answer_reward"] + rewards["format_reward"],
    }

  answer_plus_format_reward_fn.__name__ = f"{reward_fn.__name__}_answer_plus_format"
  return answer_plus_format_reward_fn


def is_strict_r1_format(response: str) -> bool:
  return re.fullmatch(r"\s*.+?</think> <answer>.+?</answer>\s*", response, flags=re.DOTALL) is not None


def with_reward_format_mode(prompt_name: str, reward_fn: RewardFn, reward_format_mode: RewardFormatMode) -> RewardFn:
  if reward_format_mode == "loose":
    return reward_fn
  if reward_format_mode != "strict":
    raise ValueError(f"Unknown reward format mode: {reward_format_mode}")
  if prompt_name == "question_only":
    return reward_fn

  def strict_format_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    rewards = reward_fn(response, ground_truth)
    if is_strict_r1_format(response):
      return rewards
    return {
      **rewards,
      "format_reward": 0.0,
      "answer_reward": 0.0,
      "reward": 0.0,
    }

  strict_format_reward_fn.__name__ = f"{reward_fn.__name__}_strict_format"
  return strict_format_reward_fn


def extract_r1_answer(response: str) -> str | None:
  if "<answer>" not in response or "</answer>" not in response:
    return None
  answer = response.split("<answer>")[-1].split("</answer>", maxsplit=1)[0].strip()
  if "\\boxed" in answer:
    return extract_answer(answer)
  return answer or None


PROMPT_EXTRACT_FNS: dict[str, ExtractFn] = {
  "question_only": extract_answer,
  "r1_zero": extract_r1_answer,
  "r1_zero_three_shot_gsm8k": extract_r1_answer,
}


def load_prompt_template(name: str) -> str:
  with open(PROMPTS_DIR / f"{name}.prompt") as f:
    return f.read()


def prompt_reward_fn(name: str, reward_mode: RewardMode, reward_format_mode: RewardFormatMode) -> RewardFn:
  reward_fn = with_reward_format_mode(name, PROMPT_REWARD_FNS[name], reward_format_mode)
  return with_reward_mode(reward_fn, reward_mode)


def load_prompts(reward_mode: RewardMode = "answer", reward_format_mode: RewardFormatMode = "loose") -> list[Prompt]:
  return [
    Prompt(
      name=name,
      template=load_prompt_template(name),
      extract=PROMPT_EXTRACT_FNS[name],
      reward_fn=prompt_reward_fn(name, reward_mode, reward_format_mode),
    )
    for name in PROMPT_KINDS
  ]


def get_prompt(name: str, reward_mode: RewardMode = "answer", reward_format_mode: RewardFormatMode = "loose") -> Prompt:
  if name not in PROMPT_KINDS:
    raise ValueError(f"Unknown prompt kind: {name}")
  return Prompt(
    name=name,
    template=load_prompt_template(name),
    extract=PROMPT_EXTRACT_FNS[name],
    reward_fn=prompt_reward_fn(name, reward_mode, reward_format_mode),
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


def make_smoke_rollouts(
  prompt: Prompt,
  examples: list[dict[str, str]],
  group_size: int,
) -> tuple[list[str], list[str], list[str]]:
  # if group_size < 2:
  #   raise ValueError("Use group_size >= 2 so each prompt has a correct and incorrect rollout.")

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
    rollout_outputs = rollout_outputs[:group_size]

    prompts.extend([prompt.format(question=example["question"])] * group_size)
    outputs.extend(rollout_outputs)
    ground_truths.extend([answer] * group_size)

  return prompts, outputs, ground_truths


def make_vllm_rollouts(
  base_url: str | VLLMServer,
  model_id: str,
  prompt: Prompt,
  examples: list[dict[str, str]],
  group_size: int,
  temperature: float,
  max_tokens: int,
  seed: int,
  batch_size: int | None,
) -> tuple[list[str], list[str], list[str]]:
  from cs336_alignment.vllm_utils import generate_completions

  prompt_strs = [prompt.format(question=example["question"]) for example in examples]
  sampling_params = {
    "temperature": temperature,
    "max_tokens": max_tokens,
    "n": group_size,
    "seed": seed,
  }
  if isinstance(base_url, VLLMServer):
    completions = base_url.generate_completions(
      prompt_strs,
      sampling_params,
      batch_size,
    )
  else:
    completions = generate_completions(
      base_url,
      model_id,
      prompt_strs,
      sampling_params,
      batch_size,
    )
  if len(completions) != len(examples) * group_size:
    raise RuntimeError(f"Expected {len(examples) * group_size} completions, got {len(completions)}")

  repeated_prompts: list[str] = []
  outputs: list[str] = []
  repeated_ground_truths: list[str] = []
  for example_index, example in enumerate(examples):
    start = example_index * group_size
    end = start + group_size
    repeated_prompts.extend([prompt_strs[example_index]] * group_size)
    outputs.extend(completion.text for completion in completions[start:end])
    repeated_ground_truths.extend([example["answer"]] * group_size)

  return repeated_prompts, outputs, repeated_ground_truths
