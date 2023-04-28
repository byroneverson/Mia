# Requires peft from git
# pip install git+https://github.com/huggingface/peft.git

import torch
import peft
from peft import PeftModel
import transformers
import os, time
import tempfile
import json
import shutil

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

BASE_MODEL = "decapoda-research/llama-7b-hf" #"./llama-7b-hf"
BASE_MODEL_DIR = "./" + BASE_MODEL
PEFT_ADAPTER_SE = "trl-lib/llama-7b-se-peft"
PEFT_ADAPTER_SE_DIR = "./" + PEFT_ADAPTER_SE
PEFT_ADAPTER_RL = "trl-lib/llama-7b-se-rl-peft"
PEFT_ADAPTER_RL_DIR = "./" + PEFT_ADAPTER_RL
MERGED_MODEL_SE = "./llama-7b-se"
MERGED_MODEL_RL = "./llama-7b-se-rl"

print("Loading base llama model")
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    cache_dir=BASE_MODEL_DIR,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print("Loading se adapter")
model = PeftModel.from_pretrained(
    model,
    PEFT_ADAPTER_SE,
    cache_dir= PEFT_ADAPTER_SE_DIR,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print("Merging base with se")
model = model.merge_and_unload()

print("Saving merged se model")
model.save_pretrained(MERGED_MODEL_SE)

print("Removing local base model and se adapter")
shutil.rmtree(BASE_MODEL_DIR)
shutil.rmtree(PEFT_ADAPTER_SE_DIR)

print("Loading merged se model")
model = LlamaForCausalLM.from_pretrained(
    MERGED_MODEL_SE,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print("Loading se-rl adapter")
model = PeftModel.from_pretrained(
    model,
    PEFT_ADAPTER_RL,
    cache_dir= PEFT_ADAPTER_RL_DIR,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

print("Merging lora with base")
for layer in model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

model.train(False)

model_sd = model.state_dict()

params = {
    "dim": 4096,
    "multiple_of": 256,
    "n_heads": 32,
    "n_layers": 32,
    "norm_eps": 1e-06,
    "vocab_size": -1,
}
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
dims_per_head = dim // n_heads
base = 10000.0
inv_freq = 1.0 / (
    base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
)

def permute(w):
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )

def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )

def translate_state_dict_key(k):  # noqa: C901
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError

new_state_dict = {}
for k, v in model_sd.items():
    new_k = translate_state_dict_key(k)
    if new_k is not None:
        if "wq" in new_k or "wk" in new_k:
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v

os.makedirs(MERGED_MODEL_RL, exist_ok=True)

print("Saving torch model and params in " + MERGED_MODEL_RL)
torch.save(new_state_dict, MERGED_MODEL_RL + "/consolidated.00.pth")
with open(MERGED_MODEL_RL + "/params.json", "w") as f:
    json.dump(params, f)

print("Removing temp local se merged model and rl adapter")
shutil.rmtree(MERGED_MODEL_SE)
shutil.rmtree(PEFT_ADAPTER_RL_DIR)

print("Completed")
