import torch

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer

from ..checkpoint import get_model_and_tokenizer

from .config import TrainConfig


class TinyByteTokenizer(PreTrainedTokenizer):
  model_input_names = ["input_ids", "attention_mask"]
  byte_vocab_size = 256
  eos_token_id_value = 256
  pad_token_id_value = 257

  def __init__(self, **kwargs):
    self._byte_vocab = {chr(i): i for i in range(self.byte_vocab_size)}
    self._byte_vocab["<pad>"] = self.pad_token_id_value
    self._byte_vocab["<eos>"] = self.eos_token_id_value
    self._id_to_token = {i: chr(i) for i in range(self.byte_vocab_size)}
    self._id_to_token[self.pad_token_id_value] = "<pad>"
    self._id_to_token[self.eos_token_id_value] = "<eos>"
    super().__init__(
      pad_token="<pad>",
      eos_token="<eos>",
      bos_token="<eos>",
      **kwargs,
    )

  @property
  def vocab_size(self) -> int:
    return self.byte_vocab_size + 2

  def __len__(self) -> int:
    return self.vocab_size

  def get_vocab(self) -> dict[str, int]:
    return dict(self._byte_vocab)

  def _tokenize(self, text: str, **kwargs) -> list[str]:
    del kwargs
    return [chr(byte) for byte in text.encode("utf-8")]

  def _convert_token_to_id(self, token: str) -> int:
    return self._byte_vocab[token]

  def _convert_id_to_token(self, index: int) -> str:
    return self._id_to_token[index]

  def convert_tokens_to_string(self, tokens: list[str]) -> str:
    byte_values = []
    for token in tokens:
      if token in {self.eos_token, self.pad_token}:  # type: ignore
        continue
      byte_values.append(ord(token))
    return bytes(byte_values).decode("utf-8", errors="replace")

  def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
    if token_ids_1 is None:
      return token_ids_0
    return token_ids_0 + token_ids_1

  def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
    raise NotImplementedError("TinyByteTokenizer is constructed in code and does not save a vocab file.")


def tiny_byte_tokenizer() -> TinyByteTokenizer:
  return TinyByteTokenizer()


def tiny_train_model(tokenizer, n_positions: int = 1024, device=None):
  torch.manual_seed(0)
  vocab_size = len(tokenizer)
  eos_token_id = tokenizer.eos_token_id
  pad_token_id = tokenizer.pad_token_id

  config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=n_positions,
    n_ctx=n_positions,
    n_embd=8,
    n_layer=1,
    n_head=2,
    n_inner=16,
    resid_pdrop=0.0,
    embd_pdrop=0.0,
    attn_pdrop=0.0,
    use_cache=False,
    bos_token_id=eos_token_id,
    eos_token_id=eos_token_id,
    pad_token_id=pad_token_id,
  )
  model = GPT2LMHeadModel(config)
  model.train()
  return model.to(device or available_device())  # type: ignore

def available_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
  return device


def build_model_and_tokenizer(config: TrainConfig, device: torch.device):
  if config.model == "tiny":
    tokenizer = tiny_byte_tokenizer()
    model = tiny_train_model(tokenizer, device=device)
    return model, tokenizer
  if config.model == "olmo2-1B":
    model_path = config.model_path()
    if model_path is None:
      raise ValueError("model_path is required for olmo2-1B")
    model, tokenizer = get_model_and_tokenizer(str(model_path), str(device))
    if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token
    model.train()
    return model, tokenizer
  raise ValueError(f"Unknown model: {config.model}")


def model_context_length(model) -> int | None:
  for attr in ("n_positions", "max_position_embeddings", "seq_length"):
    value = getattr(model.config, attr, None)
    if value is not None:
      return value
  return None
