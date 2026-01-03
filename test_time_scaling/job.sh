#!/bin/bash

# llm=llama prm=qwen800 data=amc23 method=BoNlast
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py

# llm=llama prm=qwen800 data=amc23 method=beam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py

# llm=qwen prm=qwen800 data=gsm8k method=pass1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python pass1.py

# llm=qwen prm=qwen800 data=gsm8k method=pass8
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python pass8.py