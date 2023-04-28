
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

prompt = "This is a test"

input_ids = tokenizer(prompt, truncation=True)

print(input_ids)
