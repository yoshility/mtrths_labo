import json
from tqdm import tqdm
from datasets import load_dataset
import argparse

from max_max import answer_question
from prm import prm

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

def main(index):
    N = 3

    data = load_dataset("openai/gsm8k", "main", split="train")

    # generate answer
    prompt = data[index]["question"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant solving math problems."},
        {"role": "user", "content": prompt}
    ]
    final_answer = None
    max_aggregated_score = 0
    for i in range(N):
        # full answer
        answer_all = answer_question(messages, return_all=True)
        # aggregate prm scores -> min
        aggregated_score = min(prm(answer, return_all=True))
        if aggregated_score > max_aggregated_score:
            final_answer = answer_all

    # ground truth
    _gt = data[index]["answer"]
    gt = get_answer(_gt)

    # check the answer
    is_correct = check_is_correct(gt, final_answer[2]["content"])

    # save model outputs to file
    result = {
        "index": index,
        "question": prompt,
        "pred": final_answer[2]["content"],
        "answer": _gt,
        "is_correct": is_correct
    }
    with open(f"/data/yoshie/mtrths_labo/output_bon_llama3_gsm8k.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    for index in tqdm(range(2)):
        main(index=index)