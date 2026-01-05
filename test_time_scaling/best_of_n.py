import json
from tqdm import tqdm
from datasets import load_dataset

from prm import PRM
from models_default import Llama3
from utils import get_answer, check_is_correct

# TODO
dataset_name = 'mmlu' # ['gsm8k', 'aime25', 'math_qa', 'amc23', 'mmlu']

NUM_SAMPLE = 8

llm = Llama3()

prm = PRM()

if __name__ == '__main__':
    # TODO: dataset
    # data = load_dataset("openai/gsm8k", "main", split="train")
    # data = load_dataset("math-ai/amc23", split="test")
    _data = load_dataset("TIGER-Lab/MMLU-STEM", split="test")
    data = [d for d in _data if d.get("subject") == "high_school_mathematics"] # まずは高校数学から試してみる

    for i in tqdm(range(2)): # TODO
        print(f"\nProcessing question No.{i} ...\n")
        # prompt = data[i]["question"] # gsm8k, amc23
        prompt = "Please choose the correct answer of the following question from the options.\n# Question:\n" + data[i]["question"] + "\n# Options:\n" + str(data[i]["choices"]) # mmlu

        # generate answer
        answer_candidates = []
        best_answer_index = 0 # the best answer index
        max_aggregated_score = 0
        for j in tqdm(range(NUM_SAMPLE)):
            # answer
            # llm = Llama3()
            answer = llm.infer(prompt).cpu() # only new output
            answer_chat_template = [
                {"role": "system", "content": "You are a helpful assistant solving math problems."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
            # llm.release_memory()

            # ground truth TODO
            # _gt = data[i]["answer"] # gsm8k, amc23
            _gt = [data[i]["choices"], data[i]["answer"]] # mmlu
            gt = get_answer(dataset_name, _gt)

            # check the answer
            is_correct = check_is_correct(dataset_name, gt, answer)

            # aggregate prm scores -> last
            # prm = PRM()
            scores = prm.prm(answer_chat_template, return_all=True)
            if scores == None:
                aggregated_score = 0 # CUDA OOM
            else:
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
        # output_file = "/data/yoshie/mtrths_labo/output_BoNlast_llama3_qwen800_amc23.jsonl"
        output_file = "/content/output_BoNlast_llama3_qwen800_mmlu.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for j in range(NUM_SAMPLE):
                f.write(json.dumps(answer_candidates[j], ensure_ascii=False) + "\n")
