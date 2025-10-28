# Do max-max on GSM8K dataset

import json
from tqdm import tqdm
from datasets import load_dataset
import argparse

from max_max import answer_question

def get_answer(answer):
    i = -1
    ans = ""
    while (answer[i] != ' '):
        ans = answer[i] + ans
        i -= 1
    return ans

def check_is_correct(gt, pred):
    # put ',' in gt (e.g. 488000 -> 488,000)
    i = len(gt)-1
    cnt = 0
    new_gt = ""
    while i >= 0:
        if cnt != 0 and cnt % 3 == 0:
            new_gt = ',' + new_gt
        new_gt = gt[i] + new_gt
        cnt += 1
        i -= 1
    
    # get final sentence from answer
    new_pred = pred.split('\n\n')[-1]

    return int(new_gt in new_pred or gt in new_pred)

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
    answer = answer_question(messages)

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
    with open(f"/data/yoshie/mtrths_labo/output_maxmaxprm_llama3_gsm8k.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()

