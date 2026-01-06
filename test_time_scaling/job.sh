#!/bin/bash

# llm=llama prm=qwen800 data=gsm8k method=pass1 Done

# llm=llama prm=qwen800 data=gsm8k method=pass8 Done

# llm=llama prm=qwen800 data=gsm8k method=bon Done

# llm=llama prm=qwen800 data=gsm8k method=beam Done

# llm=llama prm=qwen800 data=mmlu method=pass1 Done

# llm=llama prm=qwen800 data=mmlu method=pass8 Done

# llm=llama prm=qwen800 data=mmlu method=bon
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm llama --prm qwen800 --data mmlu --start 0 --end 100

# llm=llama prm=qwen800 data=mmlu method=beam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 0 --end 100

# llm=llama prm=qwenPRM data=gsm8k method=bon
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm llama --prm qwenPRM --data gsm8k --start 0 --end 100

# llm=llama prm=qwenPRM data=gsm8k method=beam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwenPRM --data gsm8k --start 0 --end 100

# llm=llama prm=qwenPRM data=mmlu method=bon
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm llama --prm qwenPRM --data gsm8k --start 0 --end 100

# llm=llama prm=qwenPRM data=mmlu method=beam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwenPRM --data mmlu --start 0 --end 100

# llm=qwen prm=qwen800 data=gsm8k method=pass1 Done

# llm=qwen prm=qwen800 data=gsm8k method=pass8 Done

# llm=qwen prm=qwen800 data=gsm8k method=bon
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm qwen --prm qwen800 --data gsm8k --start 0 --end 100

# llm=qwen prm=qwen800 data=gsm8k method=beam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm qwen --prm qwen800 --data gsm8k --start 0 --end 100

# llm=qwen prm=qwen800 data=mmlu method=pass1 Done

# llm=qwen prm=qwen800 data=mmlu method=pass8 Now doing

# llm=qwen prm=qwen800 data=mmlu method=bon
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm qwen --prm qwen800 --data mmlu --start 0 --end 100

# llm=qwen prm=qwen800 data=mmlu method=beam
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm qwen --prm qwen800 --data mmlu --start 0 --end 100