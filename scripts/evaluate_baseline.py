# %%
from pathlib import Path
import json
import os
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
WORKSPACE_DIR = Path(__file__).parent.parent
with open(WORKSPACE_DIR / "data/gsm8k/test.jsonl") as f:
  data_gsm8k_test = [json.loads(line) for line in f.readlines()]
VLLM_BASE_URL

# %%
from dataclasses import dataclass
from cs336_alignment.rl.prompts import PROMPT_KINDS, Prompt, load_prompts

prompts = load_prompts()

# %%
from cs336_alignment.vllm_utils import VLLMServer
hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
model_id = str(hf_home / "hub" / "models--allenai--OLMo-2-0425-1B" / "snapshots" / "a1847dff35000b4271fa70afc5db10fd29fedbdf")
server = VLLMServer(model_id, gpu=0)
server.start()

# %%
from cs336_alignment.vllm_utils import generate_completions, VLLMCompletion
# VLLM_BASE_URL = "http://10.0.0.132:8008"
@dataclass
class CompletionResult:
  completion: VLLMCompletion
  reward: dict[str, float]

def run(prompt: Prompt, data: list[dict]):
  prompts = [prompt.format(question=example["question"]) for example in data]
  completions = generate_completions(
    VLLM_BASE_URL,
    "",
    prompts,
    {"temperature": 0.7, "max_tokens": 1000, "n": 1, "seed": 42},
    5,
  )
  rewards = list[CompletionResult]()
  for (i, d) in zip(completions, data):
    # answer = prompt.extract(i.text)
    reward = prompt.reward_fn(i.text, d["answer"])
    rewards.append(CompletionResult(completion=i, reward=reward))
  return rewards

# %%
import json
import os
os.makedirs(WORKSPACE_DIR / "out", exist_ok=True)
results = dict[str, list[CompletionResult]]()
for k in PROMPT_KINDS:
  p = next(p for p in prompts if p.name == k)
  rewards = run(p, data_gsm8k_test)
  results[k] = rewards
  with open(WORKSPACE_DIR / f"out/baseline_completions_{k}_test.jsonl", "w") as f:
    for r in rewards:
      f.write(json.dumps({"completion": r.completion.__dict__, "reward": r.reward}) + "\n")

# %%
"""
I found a strange thing, without prompt `\\boxed{}` prompt,
the model will continue self ask and answer like
Q: 'What is the capital of France?'
A: '\n\nThe capital of France is Paris.\n\nWhat is the capital of the United States?\n\nThe capital of the United States is Washington, D.C.\n\nWhat is the capital of Brazil?\n\nThe capital of Brazil is Brasília.\n\nWhat is'
until max lenght is reached.
"""
# generate_completions(VLLM_BASE_URL, "qwen3.6", "What is the capital of France?", {"temperature": 0.7, "max_tokens": 50, "n": 1, "seed": 42})

# %%
results = dict[str, list[CompletionResult]]()
for k in PROMPT_KINDS:
  with open(WORKSPACE_DIR / f"out/baseline_completions_{k}_test.jsonl") as f:
    rewards = [CompletionResult(completion=VLLMCompletion(**json.loads(line)["completion"]), reward=json.loads(line)["reward"]) for line in f.readlines()]
    results[k] = rewards

# %%
for k in PROMPT_KINDS:
  total_reward = {
    "answer_reward": .0,
    "format_reward": .0,
    "reward": .0,
  }
  for (i, d) in zip(results[k], data_gsm8k_test):
    reward = i.reward
    total_reward["reward"] += reward["reward"]
    total_reward["answer_reward"] += reward["answer_reward"]
    total_reward["format_reward"] += reward["format_reward"]
  print(f"Total reward for {k}: {total_reward}")

# %%
