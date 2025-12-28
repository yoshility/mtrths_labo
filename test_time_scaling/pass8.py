import json
from tqdm import tqdm
from datasets import load_dataset

from models_default import Llama3
from utils import get_answer, check_is_correct

# TODO
dataset_name = 'aime25' # ['gsm8k', 'aime25']

NUM_SAMPLE = 8

llm = Llama3()

# TODO
# data = load_dataset("openai/gsm8k", "main", split="train")
data = load_dataset("MathArena/aime_2025", split="train")

for i in tqdm(range(30)): # TODO
    # TODO
    # prompt = data[i]["question"] # gsm8k
    prompt = data[i]["problem"] # aime25

    # generate answers
    answers = []
    for _ in range(NUM_SAMPLE):
        answer = llm.infer(prompt)
        answers.append(answer)

    # ground truth
    _gt = data[i]["answer"] # gsm8k, aime25
    gt = get_answer(dataset_name, _gt)

    # check if there is any correct answer
    judge = []
    for j in range(NUM_SAMPLE):
        judge.append(check_is_correct(dataset_name, gt, answers[j]))

    # check the answer
    is_correct = int(any(judge))
    final_answer = answers[judge.index(True)] if is_correct else answers[0]

    # save model outputs to file
    result = {
        "index": i,
        "question": prompt,
        "pred": final_answer,
        "answer": _gt,
        "is_correct": is_correct
    }
    with open(f"/data/yoshie/mtrths_labo/output_pass8_llama3_aime25.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")