#!/bin/bash

# llm=llama prm=qwen800 data=gsm8k method=pass1 Done

# llm=llama prm=qwen800 data=gsm8k method=pass8 Done

# llm=llama prm=qwen800 data=gsm8k method=bon Done

# llm=llama prm=qwen800 data=gsm8k method=beam Done

# llm=llama prm=qwen800 data=mmlu method=pass1 Done

# llm=llama prm=qwen800 data=mmlu method=pass8 Done

# llm=llama prm=qwen800 data=mmlu method=bon Done
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm llama --prm qwen800 --data mmlu --start 0 --end 100

# llm=llama prm=qwen800 data=mmlu method=beam Done
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 0 --end 100

# llm=llama prm=qwenPRM data=gsm8k method=bon Done
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm llama --prm qwenPRM --data gsm8k --start 0 --end 100

# llm=llama prm=qwenPRM data=gsm8k method=beam 
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwenPRM --data gsm8k --start 0 --end 100

# llm=llama prm=qwenPRM data=mmlu method=bon
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm llama --prm qwenPRM --data gsm8k --start 0 --end 100

# llm=llama prm=qwenPRM data=mmlu method=beam
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwenPRM --data mmlu --start 0 --end 100

# llm=qwen prm=qwen800 data=gsm8k method=pass1 Done

# llm=qwen prm=qwen800 data=gsm8k method=pass8 Done

# llm=qwen prm=qwen800 data=gsm8k method=bon
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm qwen --prm qwen800 --data gsm8k --start 0 --end 100

# llm=qwen prm=qwen800 data=gsm8k method=beam
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm qwen --prm qwen800 --data gsm8k --start 0 --end 100

# llm=qwen prm=qwen800 data=mmlu method=pass1 Done

# llm=qwen prm=qwen800 data=mmlu method=pass8 Now doing

# llm=qwen prm=qwen800 data=mmlu method=bon
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python best_of_n.py --llm qwen --prm qwen800 --data mmlu --start 0 --end 100

# llm=qwen prm=qwen800 data=mmlu method=beam
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm qwen --prm qwen800 --data mmlu --start 0 --end 100


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 4 --end 5 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index4.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 7 --end 8 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index7.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 9 --end 10 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index9.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 19 --end 20 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index19.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 45 --end 46 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index45.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 55 --end 56 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index55.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 74 --end 75 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index74.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 82 --end 83 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index82.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 90 --end 91 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index90.txt"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python beam.py --llm llama --prm qwen800 --data mmlu --start 94 --end 95 > "/data/yoshie/mtrths_labo/debug_beam_llama_qwen800_mmlu_index94.txt"