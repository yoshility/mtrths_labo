import json
from tqdm import tqdm
from datasets import load_dataset
import argparse

from pass1 import Llama3
from prm import PRM

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


if __name__ == '__main__':
    data = load_dataset("openai/gsm8k", "main", split="train")
    # なんか急にデータ数が7473に増えた

    for i in tqdm(range(4, 100)):
        prompt = data[i]["question"]

        # generate answer
        final_answer = None # the best answer
        final_answer_scores = None # stepwise prm scores of the final answer
        max_aggregated_score = 0
        N = 3
        for _ in range(N):
            # answer
            llm = Llama3()
            answer = llm.infer(prompt) # only new output
            answer_chat_template = [
                {"role": "system", "content": "You are a helpful assistant solving math problems."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
            llm.release_memory()
            # aggregate prm scores -> min
            prm = PRM()
            scores = prm.prm(answer_chat_template, return_all=True)
            aggregated_score = min(scores)
            prm.release_memory()
            if aggregated_score > max_aggregated_score:
                max_aggregated_score = aggregated_score
                final_answer_scores = scores
                final_answer = answer   

        # ground truth
        _gt = data[i]["answer"]
        gt = get_answer(_gt)

        # check the answer
        is_correct = check_is_correct(gt, final_answer)

        # save model outputs to file
        result = {
            "index": i,
            "question": prompt,
            "pred": final_answer,
            "scores": final_answer_scores,
            "answer": _gt,
            "is_correct": is_correct
        }
        with open(f"/data/yoshie/mtrths_labo/output_BoN_llama3_gsm8k.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
