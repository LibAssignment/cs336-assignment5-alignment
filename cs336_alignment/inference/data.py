from transformers import PreTrainedTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Callable, Literal
import torch

@dataclass
class TokenizedPromptAndOutput:
  input_ids: torch.Tensor # shape (batch_size, seq_length)
  labels: torch.Tensor # shape (batch_size, seq_length)
  response_mask: torch.Tensor # shape (batch_size, seq_length)

def tokenize_prompt_and_output(
  prompt_strs: list[str],
  output_strs: list[str],
  tokenizer: PreTrainedTokenizer,
) -> TokenizedPromptAndOutput:
  # input_strs = [f"{a}{b}" for (a, b) in zip(prompt_strs, output_strs)]
  prompt_result = tokenizer(
    prompt_strs,
    padding=True,
    truncation=True,
    return_tensors="pt",
  )
  prompt_ids = prompt_result.input_ids
  prompt_lengths = prompt_result.attention_mask.sum(dim=1)
  output_result = tokenizer(
    output_strs,
    padding=True,
    truncation=True,
    return_tensors="pt",
  )
  output_ids = output_result.input_ids
  output_lengths = output_result.attention_mask.sum(dim=1)

  max_length = (prompt_lengths + output_lengths).max()
  tokenized_ids = torch.full((len(prompt_strs), max_length), tokenizer.pad_token_id)
  mask = torch.zeros_like(tokenized_ids, dtype=torch.bool)
  for i in range(len(prompt_strs)):
    tokenized_ids[i, :prompt_lengths[i]] = prompt_ids[i, :prompt_lengths[i]]
    tokenized_ids[i, prompt_lengths[i]:prompt_lengths[i]+output_lengths[i]] = output_ids[i, :output_lengths[i]]
    mask[i, prompt_lengths[i]:prompt_lengths[i]+output_lengths[i]] = True
  labels = tokenized_ids[:, 1:]
  input_ids = tokenized_ids[:, :-1]
  mask = mask[:, 1:]

  return TokenizedPromptAndOutput(input_ids=input_ids, labels=labels, response_mask=mask)


@dataclass
class ResponseLogProbs:
  log_probs: torch.Tensor
  token_entropy: torch.Tensor | None = None

def get_response_log_probs(
  model: PreTrainedModel,
  input_ids: torch.Tensor, # shape (batch_size, seq_length)
  labels: torch.Tensor, # shape (batch_size, seq_length)
  return_token_entropy: bool = False,
) -> ResponseLogProbs:
  with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)
    output_logits = outputs.logits
    # print(output_logits.shape, labels.shape)
    log_probs = -torch.nn.functional.cross_entropy(
      output_logits.view(-1, output_logits.size(-1)),
      labels.view(-1),
      reduction="none"
    ).view(labels.size())
    # print(log_probs.shape)
    token_entropy = None
    if return_token_entropy:
      probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
      token_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return ResponseLogProbs(log_probs=log_probs, token_entropy=token_entropy)


@dataclass
class RolloutRewards:
  raw_rewards: torch.Tensor # shape (batch_size,)
  metadata: dict[str, float] | None = None


def compute_rollout_rewards(
  reward_fn: Callable[[str, str], dict[str, float]],
  rollout_responses: list[str],
  repeated_ground_truths: list[str],
) -> RolloutRewards:
  rewards = [reward_fn(response, ground_truth) for (response, ground_truth) in zip(rollout_responses, repeated_ground_truths)]
  raw_rewards = torch.tensor([reward["reward"] for reward in rewards])
  keys = rewards[0].keys() if len(rewards) > 0 else []
  metadata = {key: sum([reward[key] for reward in rewards])/len(rewards) for key in keys}
  return RolloutRewards(raw_rewards=raw_rewards, metadata=metadata)


@dataclass
class GroupRewards:
  advantages: torch.Tensor # shape (batch_size,)
  metadata: dict[str, float] | None = None


def compute_group_normalized_rewards(
  raw_rewards: torch.Tensor, # shape (batch_size,)
  group_size: int,
  baseline: Literal["mean", "none"] = "mean",
  advantage_eps: float = 1e-8,
  advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> GroupRewards:
  rewards = raw_rewards.view(-1, group_size)
  baseline_value = torch.zeros(rewards.size(0))
  if baseline == "mean":
    baseline_value = rewards.mean(dim=1)
  advantage_norm = torch.ones_like(baseline_value)
  if advantage_normalizer == "std":
    advantage_norm = rewards.std(dim=1) + advantage_eps
  elif advantage_normalizer == "mean":
    advantage_norm = rewards.mean(dim=1) + advantage_eps
  advantages = (rewards - baseline_value[:, None]) / advantage_norm[:, None]
  return GroupRewards(advantages=advantages.view(-1), metadata={
    "baseline_value": baseline_value.mean().item(),
    "advantage_norm": advantage_norm.mean().item(),
  })
