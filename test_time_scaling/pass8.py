import json
from tqdm import tqdm
from datasets import load_dataset

from models_default import Llama3, Qwen
from utils import get_answer, check_is_correct

# TODO
dataset_name = 'gsm8k' # ['gsm8k', 'aime25', 'math_qa', 'amc23]

NUM_SAMPLE = 8

# TODO
# llm = Llama3()
llm = Qwen()

# TODO: dataset
data = load_dataset("openai/gsm8k", "main", split="train")
# data = load_dataset("MathArena/aime_2025", split="train")
# data = load_dataset("allenai/math_qa", split="train", trust_remote_code=True)
# data = load_dataset("math-ai/amc23", split="test")

for i in tqdm(range(1, 100)): # TODO
    # TODO
    prompt = data[i]["question"] # gsm8k, amc23
    # prompt = data[i]["problem"] # aime25
    # prompt = "Please choose the correct answer of the following question from the options.\n# Question:\n" + data[i]["Problem"] + "\n# Options:\n" + data[i]["options"] # math_qa

    # generate answers
    answers = []
    for _ in range(NUM_SAMPLE):
        answer = llm.infer(prompt)
        answers.append(answer)

    # ground truth TODO
    _gt = data[i]["answer"] # gsm8k, aime25, amc23
    # _gt = [data[i]["options"], data[i]["correct"]] # math_qa
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
    with open(f"/data/yoshie/mtrths_labo/output_pass8_qwen_gsm8k.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")