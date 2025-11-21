import json
from tqdm import tqdm
from datasets import load_dataset

from pass1 import Llama3
from prm import PRM
from utils import get_answer, check_is_correct

NUM_SAMPLE = 8

if __name__ == '__main__':
    data = load_dataset("openai/gsm8k", "main", split="train")
    # なんか急にデータ数が7473に増えた

    for i in tqdm(range(100)):
        prompt = data[i]["question"]

        # generate answer
        final_answer = None # the best answer
        final_answer_scores = None # stepwise prm scores of the final answer
        max_aggregated_score = 0
        for _ in range(NUM_SAMPLE):
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
            aggregated_score = scores[-1] # BoN_Last
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
        with open(f"/data/yoshie/mtrths_labo/output_BoNlast_llama3_gsm8k.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
