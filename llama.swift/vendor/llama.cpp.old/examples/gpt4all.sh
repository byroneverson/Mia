#!/bin/bash

#
# Temporary script - will be removed in the future
#

cd `dirname $0`
cd ..

./main -m ./models/gpt4all-lora-quantized.bin --color -f ./prompts/alpaca.txt -ins -b 256 --top_k 10000 --temp 0.35 --repeat_penalty 1 -t 7
