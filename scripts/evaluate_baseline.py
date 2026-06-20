# %%
from pathlib import Path
import sys
import json
import os
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
WORKSPACE_DIR = Path(__file__).parent.parent
# sys.path.append(str(WORKSPACE_DIR/"cs336_alignment"))

with open(WORKSPACE_DIR / "cs336_alignment/prompts/question_only.prompt") as f:
  prompt_q = f.read()
with open(WORKSPACE_DIR / "data/gsm8k/test.jsonl") as f:
  data_gsm8k_test = [json.loads(line) for line in f.readlines()]
prompt_q, VLLM_BASE_URL

# %%
from cs336_alignment.vllm_utils import VLLMServer
model_id = os.path.expanduser("~/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/a1847dff35000b4271fa70afc5db10fd29fedbdf")
server = VLLMServer(model_id, gpu=0)
server.start()

# %%
from cs336_alignment.vllm_utils import generate_completions
# VLLM_BASE_URL = "http://10.0.0.132:8008"
prompts = [prompt_q.format(question=example["question"]) for example in data_gsm8k_test]
completions = generate_completions(
  VLLM_BASE_URL,
  "",
  prompts,
  {"temperature": 0.7, "max_tokens": 1000, "n": 1, "seed": 42},
  5,
)
completions

# %%
import json
import os
os.makedirs(WORKSPACE_DIR / "out", exist_ok=True)
with open(WORKSPACE_DIR / "out/baseline_completions_questions_only_test.json", "w") as f:
  for i in completions:
    f.write(json.dumps(i.__dict__) + "\n")


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
from cs336_alignment.drgrpo_grader import question_only_reward_fn
total_reward = {
  "answer_reward": .0,
  "format_reward": .0,
  "reward": .0,
}
for (i, d) in zip(completions, data_gsm8k_test):
  reward = question_only_reward_fn(i.text, d["answer"])
  total_reward["reward"] += reward["reward"]
  total_reward["answer_reward"] += reward["answer_reward"]
  total_reward["format_reward"] += reward["format_reward"]
total_reward


# %%
