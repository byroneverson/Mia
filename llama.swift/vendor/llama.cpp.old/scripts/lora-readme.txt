ggml lora recap - be sure to read all scripts to ensure correct params are provided

1. convert orig llama weights to hf (convert_llama_weights_to_hf.py)
2. merge hf weights with hf peft lora adapter weights, outputs in correct pth format (merge_llama_hf_to_peft.py)
3. copy consolidated pth to 7B folder
4. convert merged pth to ggml f16 (convert-pth-to-ggml.py)
5. quantize f16 to q0 (quantize.py)