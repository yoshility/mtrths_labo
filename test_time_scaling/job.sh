#!/bin/bash

# llm=llama prm=qwen800 data=amc23 method=beam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py

# llm=llama prm=qwenPRM data=gsm8k method=BoNlast
