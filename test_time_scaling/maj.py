'''
答え抽出がめんどいので一旦廃止
'''

import re
import json
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter

from pass1 import Llama3
from utils import get_answer

NUM_SAMPLE = 8

if __name__ == '__main__':
    llm = Llama3()
    data = load_dataset("openai/gsm8k", "main", split="train")
    
    for i in tqdm(range(100)):
        prompt = data[i]["question"]

        # generate answers
        answers = []
        for _ in range(NUM_SAMPLE):
            answer = llm.infer(prompt)
            answers.append(answer)

        # majority vote
        nums = []
        for j in range(NUM_SAMPLE):
            final_step = answers[j].split('\n\n')[-1]
            match = re.findall(r'[-+]?\d[\d,]*\.?\d*', final_step)
            num_str = match[-1].replace(',', '')
            nums.append(num_str)
        final_num_answer = Counter(nums).most_common(1)[0][0]
        final_answer = answers[nums.index(final_num_answer)]
        
        # ground truth
        _gt = data[i]["answer"]
        gt = get_answer(_gt)

        # check the answer
        is_correct = int(gt == final_num_answer)

        # save model outputs to file
        result = {
            "index": i,
            "question": prompt,
            "candidates": nums,
            "pred": final_answer,
            "answer": _gt,
            "is_correct": is_correct
        }
        with open(f"/data/yoshie/mtrths_labo/output_maj_llama3_gsm8k.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
