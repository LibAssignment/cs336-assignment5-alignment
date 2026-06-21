from transformers import PreTrainedTokenizer, PreTrainedModel
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
  input_ids: torch.Tensor,
  labels: torch.Tensor,
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
