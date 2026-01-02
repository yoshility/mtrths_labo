import json
from tqdm import tqdm
from datasets import load_dataset

from prm import PRM
from models_default import Llama3
from utils import get_answer, check_is_correct

NUM_SAMPLE = 8

if __name__ == '__main__':
    data = load_dataset("openai/gsm8k", "main", split="train")
    # なんか急にデータ数が7473に増えた

    for i in tqdm(range(4, 100)):
        print(f"\nProcessing question No.{i} ...\n")
        prompt = data[i]["question"]

        # generate answer
        answer_candidates = []
        best_answer_index = None # the best answer index
        max_aggregated_score = 0
        for j in tqdm(range(NUM_SAMPLE)):
            # answer
            llm = Llama3()
            answer = llm.infer(prompt) # only new output
            answer_chat_template = [
                {"role": "system", "content": "You are a helpful assistant solving math problems."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
            llm.release_memory()

            # ground truth
            _gt = data[i]["answer"]
            gt = get_answer(_gt)

            # check the answer
            is_correct = check_is_correct(gt, answer)

            # aggregate prm scores -> last
            prm = PRM()
            scores = prm.prm(answer_chat_template, return_all=True)
            aggregated_score = scores[-1] # BoN_Last
            prm.release_memory()

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
        with open(f"/data/yoshie/mtrths_labo/output_BoNlast_llama3_qwen800_gsm8k_allCandidates.jsonl", "a", encoding="utf-8") as f:
            for j in range(NUM_SAMPLE):
                f.write(json.dumps(answer_candidates[j], ensure_ascii=False) + "\n")
