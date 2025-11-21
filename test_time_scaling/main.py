# Do max-max on GSM8K dataset

import json
from datasets import load_dataset
import argparse

from max_max import answer_question
from utils import get_answer, check_is_correct

def main():
    data = load_dataset("openai/gsm8k", "main", split="train")

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int)
    args = parser.parse_args()

    # generate answer
    prompt = data[args.index]["question"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant solving math problems."},
        {"role": "user", "content": prompt}
    ]
    answer = answer_question(messages, return_all=False)

    # ground truth
    _gt = data[args.index]["answer"]
    gt = get_answer(_gt)

    # check the answer
    is_correct = check_is_correct(gt, answer)

    # save model outputs to file
    result = {
        "index": args.index,
        "question": prompt,
        "pred": answer,
        "answer": _gt,
        "is_correct": is_correct
    }
    with open(f"/data/yoshie/mtrths_labo/output_maxmaxprm_llama3_gsm8k_ver2.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    # print(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()

