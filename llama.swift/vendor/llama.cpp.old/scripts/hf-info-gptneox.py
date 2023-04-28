# Hugging Face fine-tuned gpt-neox-like models info

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

if len(sys.argv) < 2:
    print("Usage: python hf-info-gptneox.py model_name")
    print("  model_name: name of the model. Example: 'OpenAssistant/stablelm-7b-sft-v7-epoch-3'")
    print("  cache-dir-prefix: prefix of stored model cache directory. Example: 'stablelm'")
    sys.exit(1)

model_name = sys.argv[1]
dir_out = sys.argv[2]
model_cache_dir = dir_out + "-cache"

tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading model: ", model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=model_cache_dir)
model.eval()
for p in model.parameters():
    p.requires_grad = False
hparams = model.config.to_dict()
print("Model loaded: ", model_name)
print("")
print(hparams)
print("")
list_vars = model.state_dict()
print(list_vars)
print("")
print("Done.")
print("")
