import torch

from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

def tiny_train_model(tokenizer):
  torch.manual_seed(0)
  config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=16,
    n_ctx=16,
    n_embd=8,
    n_layer=1,
    n_head=2,
    n_inner=16,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_cache=False,
    bos_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
  )
  model = GPT2LMHeadModel(config)
  model.train()
  return model.to(available_device())

def available_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
  return device
