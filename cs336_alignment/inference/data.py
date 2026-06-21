from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import torch

@dataclass
class TokenizedPromptAndOutput:
  input_ids: torch.Tensor
  labels: torch.Tensor
  response_mask: torch.Tensor

def tokenize_prompt_and_output(
  prompt_strs: list[str],
  output_strs: list[str],
  tokenizer: PreTrainedTokenizer,
) -> TokenizedPromptAndOutput:
  # input_strs = [f"{a}{b}" for (a, b) in zip(prompt_strs, output_strs)]
  prompt_ids = tokenizer(
    prompt_strs,
    padding=True,
    truncation=True,
    return_tensors="pt",
  ).input_ids
  output_ids = tokenizer(
    output_strs,
    padding=True,
    truncation=True,
    return_tensors="pt",
  ).input_ids

  prompt_lengths = (prompt_ids != tokenizer.pad_token_id).sum(dim=1)
  output_lengths = (output_ids != tokenizer.pad_token_id).sum(dim=1)
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
