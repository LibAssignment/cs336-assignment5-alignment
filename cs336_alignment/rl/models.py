import torch

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer


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
      if token in {self.eos_token, self.pad_token}:
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


def tiny_train_model(tokenizer=None, vocab_size: int | None = None, n_positions: int = 256, device=None):
  torch.manual_seed(0)
  if tokenizer is not None:
    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
  else:
    vocab_size = vocab_size or TinyByteTokenizer.byte_vocab_size + 2
    pad_token_id = TinyByteTokenizer.pad_token_id_value
    eos_token_id = TinyByteTokenizer.eos_token_id_value

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
  return model.to(device or available_device())

def available_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
  elif torch.backends.mps.is_available():
    device = torch.device("mps")
  else:
    device = torch.device("cpu")
  return device
