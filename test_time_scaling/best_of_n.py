import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

from models_default import Llama3, Qwen
from prm import PRM, QwenPRM
from utils import get_answer, check_is_correct

NUM_SAMPLE = 8

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm') # ['llama', 'qwen']
    parser.add_argument('--prm') # ['qwen800', 'qwenPRM']
    parser.add_argument('--data') # ['gsm8k', 'mmlu']
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()

    # LLM
    if args.llm == 'llama':
        llm = Llama3()
    elif args.llm == 'qwen':
        llm = Qwen()

    # PRM
    if args.prm == 'qwen800':
        prm = PRM()
        # prm = Qwen800()
    elif args.prm == 'qwenPRM':
        prm = QwenPRM()

    # dataset
    if args.data == 'gsm8k':
        data = load_dataset("openai/gsm8k", "main", split="train")
        dataset_name = 'gsm8k'
    elif args.data == 'mmlu':
        _data = load_dataset("TIGER-Lab/MMLU-STEM", split="test")
        data = [d for d in _data if d.get("subject") == "high_school_mathematics"]
        dataset_name = 'mmlu'

    for i in tqdm(range(args.start, args.end)):
        if args.data == 'gsm8k':
            prompt = data[i]["question"] # gsm8k
        elif args.data == 'mmlu':
            prompt = "Please choose the correct answer of the following question from the options.\n# Question:\n" + data[i]["question"] + "\n# Options:\n" + str(data[i]["choices"]) # mmlu

        # generate answer
        answer_candidates = []
        best_answer_index = 0 # the best answer index
        max_aggregated_score = 0
        for j in tqdm(range(NUM_SAMPLE)):
            # answer
            # llm = Llama3()
            answer = llm.infer(prompt) # only new output
            answer_chat_template = [
                {"role": "system", "content": "You are a helpful assistant solving math problems. Solve the problem step by step. Separate the steps with '\\n\\n' to make it readable."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
            # llm.release_memory()

            # ground truth
            if args.data == 'gsm8k':
                _gt = data[i]["answer"] # gsm8k
            elif args.data == 'mmlu':
                _gt = [data[i]["choices"], data[i]["answer"]] # mmlu
            gt = get_answer(dataset_name, _gt)

            # check the answer
            is_correct = check_is_correct(dataset_name, gt, answer)

            # aggregate prm scores -> last
            # prm = PRM()
            scores = prm.prm(answer_chat_template, return_all=True)
            aggregated_score = scores[-1] # BoN_Last
            # prm.release_memory()

            # best of N
            if aggregated_score > max_aggregated_score:
                max_aggregated_score = aggregated_score
                best_answer_index = j   

            # candidate answer data
            result = {
                "index": i,
                "candidate": j,
                "question": prompt,
                "pred": answer,
                "scores": scores,
                "answer": _gt,
                "is_correct": is_correct,
                "is_best": 0 # temporary 0
            }
            answer_candidates.append(result)
        
        # best answer
        answer_candidates[best_answer_index]["is_best"] = 1
        
        # save all the answers
        output_file = f"/data/yoshie/mtrths_labo/output_BoNlast_{args.llm}_{args.prm}_{args.data}.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for j in range(NUM_SAMPLE):
                f.write(json.dumps(answer_candidates[j], ensure_ascii=False) + "\n")
