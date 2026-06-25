from .utils import *
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class GRPOTrainStepResult:
  loss: torch.Tensor
  log_probs: torch.Tensor | None
  advantages: torch.Tensor
  rewards: torch.Tensor
  answer_lengths: torch.Tensor
  seq_perplexity: float
  answer_perplexity: float
  reward_metadata: dict[str, float] | None = None

  def metadata(self) -> dict[str, torch.Tensor]:
    return {
      **({} if self.log_probs is None else {"log_probs": self.log_probs.detach().cpu()}),
      "advantages": self.advantages.detach().cpu(),
      "rewards": self.rewards.detach().cpu(),
    }

def grpo_train_step(
  model: PreTrainedModel,
  tokenizer: PreTrainedTokenizer,
  optimizer: torch.optim.Optimizer,
  reward_fn: Callable[[str, str], dict[str, float]],
  prompt_strs: list[str],
  output_strs: list[str],
  ground_truths: list[str],
  group_size: int,

  macrobatch_size: int,
  max_grad_norm: float | None = None,
  returns_log_probs: bool = False,

  # compute_group_normalized_rewards
  baseline: Literal["mean", "none"] = "mean",
  advantage_eps: float = 1e-8,
  advantage_normalizer: Literal["std", "none", "mean"] = "std",

  # compute_policy_gradient_loss
  importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
  old_log_probs: torch.Tensor | None = None,
  cliprange: float | None = None,

  # aggregate_loss_across_microbatch
  loss_normalization: Literal["sequence", "constant"] = "sequence",
  normalization_constant: int | None = None,
):
  if macrobatch_size <= 0:
    raise ValueError(f"macrobatch_size must be positive, got {macrobatch_size}")

  device = model.device
  tokenized = tokenize_prompt_and_output(
    prompt_strs,
    output_strs,
    tokenizer,
  )
  x = tokenized.input_ids
  y = tokenized.labels
  rewards = compute_rollout_rewards(
    reward_fn,
    output_strs,
    ground_truths,
  )
  advantages = compute_group_normalized_rewards(
    rewards.raw_rewards,
    group_size,
    baseline=baseline,
    advantage_eps=advantage_eps,
    advantage_normalizer=advantage_normalizer,
  )

  final_loss = torch.tensor(0.0, device=device)
  final_log_probs = []
  seq_log_prob_sum = torch.tensor(0.0, device=device)
  seq_token_count = torch.tensor(0, device=device)
  answer_log_prob_sum = torch.tensor(0.0, device=device)
  answer_token_count = torch.tensor(0, device=device)
  for i in range(0, len(prompt_strs), macrobatch_size):
    x_batch = x[i:i+macrobatch_size].to(device)
    y_batch = y[i:i+macrobatch_size].to(device)
    mask_batch = tokenized.response_mask[i:i+macrobatch_size].to(device)
    seq_mask_batch = y_batch != tokenizer.pad_token_id
    attention_mask_batch = x_batch != tokenizer.pad_token_id
    adv_batch = advantages.advantages[i:i+macrobatch_size].to(device)
    old_log_probs_batch = old_log_probs[i:i+macrobatch_size].to(device) if old_log_probs is not None else None
    log_probs = get_response_log_probs(model, x_batch, y_batch, mask=attention_mask_batch, return_token_entropy=True)
    seq_log_prob_sum += (log_probs.log_probs * seq_mask_batch).sum().detach()
    seq_token_count += seq_mask_batch.sum().detach()
    answer_log_prob_sum += (log_probs.log_probs * mask_batch).sum().detach()
    answer_token_count += mask_batch.sum().detach()
    loss_batch = compute_policy_gradient_loss(
      adv_batch,
      log_probs.log_probs,
      importance_reweighting_method=importance_reweighting_method,
      old_log_probs=old_log_probs_batch,
      cliprange=cliprange,
      response_mask=mask_batch
    )
    loss = aggregate_loss_across_microbatch(
      loss_batch.per_token_policy_gradient_loss,
      mask_batch,
      loss_normalization=loss_normalization,
      normalization_constant=normalization_constant,
    )
    loss.backward()

    final_loss += loss.detach() * (x_batch.size(0) / len(prompt_strs)) # TODO: detach here to reduce GPU memory usage
    if returns_log_probs:
      final_log_probs.append(log_probs.log_probs.detach().cpu())
  if max_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
  optimizer.step()
  optimizer.zero_grad()

  return GRPOTrainStepResult(
    loss=final_loss,
    log_probs=torch.cat(final_log_probs, dim=0) if final_log_probs else None,
    advantages=advantages.advantages.detach().cpu(),
    rewards=rewards.raw_rewards.detach().cpu(),
    answer_lengths=tokenized.response_mask.sum(dim=1).detach().cpu(),
    seq_perplexity=torch.exp(-seq_log_prob_sum / seq_token_count.clamp(min=1)).item(),
    answer_perplexity=torch.exp(-answer_log_prob_sum / answer_token_count.clamp(min=1)).item(),
    reward_metadata=rewards.metadata,
  )
