from .utils import *
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass

@dataclass
class GRPOTrainStepResult:
  loss: torch.Tensor
  log_probs: torch.Tensor
  advantages: torch.Tensor
  rewards: torch.Tensor

  def metadata(self) -> dict[str, torch.Tensor]:
    return {
      "log_probs": self.log_probs.detach(),
      "advantages": self.advantages.detach(),
      "rewards": self.rewards.detach(),
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

  gradient_accumulation_steps: int,
  max_grad_norm: float | None = None,

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

  macrobatch_size = (len(prompt_strs) - 1) // gradient_accumulation_steps + 1

  final_loss = torch.tensor(0.0, device=device)
  final_log_probs = []
  for i in range(0, len(prompt_strs), macrobatch_size):
    x_batch = x[i:i+macrobatch_size].to(device)
    y_batch = y[i:i+macrobatch_size].to(device)
    mask_batch = tokenized.response_mask[i:i+macrobatch_size].to(device)
    adv_batch = advantages.advantages[i:i+macrobatch_size].to(device)
    old_log_probs_batch = old_log_probs[i:i+macrobatch_size].to(device) if old_log_probs is not None else None
    log_probs = get_response_log_probs(model, x_batch, y_batch, mask=mask_batch, return_token_entropy=True)
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

    final_loss += loss * (x_batch.size(0) / len(prompt_strs))
    final_log_probs.append(log_probs.log_probs.detach())
  if max_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
  optimizer.step()
  optimizer.zero_grad()

  return GRPOTrainStepResult(
    loss=final_loss,
    log_probs=torch.cat(final_log_probs, dim=0),
    advantages=advantages.advantages.detach(),
    rewards=rewards.raw_rewards.detach(),
  )
