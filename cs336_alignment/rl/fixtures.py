import re


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


def gsm8k_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
  parsed = extract_gsm8k_answer(response)
  answer_reward = float(parsed == ground_truth)
  format_reward = float("####" in response)
  return {
    "reward": answer_reward,
    "format_reward": format_reward,
    "answer_reward": answer_reward,
  }
