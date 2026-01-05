import json
from tqdm import tqdm
from datasets import load_dataset

from models_default import Llama3, Qwen
from utils import get_answer, check_is_correct

# TODO
dataset_name = 'mmlu' # ['gsm8k', 'aime25', 'mathqa', 'amc23', 'mmlu']

NUM_SAMPLE = 8

# TODO
llm = Llama3()
# llm = Qwen()

# TODO: dataset
# data = load_dataset("openai/gsm8k", "main", split="train")
# data = load_dataset("MathArena/aime_2025", split="train")
# data = load_dataset("allenai/math_qa", split="train", trust_remote_code=True)
# data = load_dataset("math-ai/amc23", split="test")
_data = load_dataset("TIGER-Lab/MMLU-STEM", split="test")
data = [d for d in _data if d.get("subject") == "high_school_mathematics"] # まずは高校数学から試してみる

for i in tqdm(range(11, 100)): # TODO
    # TODO
    # prompt = data[i]["question"] # gsm8k, amc23
    # prompt = data[i]["problem"] # aime25
    # prompt = "Please choose the correct answer of the following question from the options.\n# Question:\n" + data[i]["Problem"] + "\n# Options:\n" + data[i]["options"] # math_qa
    prompt = "Please choose the correct answer of the following question from the options.\n# Question:\n" + data[i]["question"] + "\n# Options:\n" + str(data[i]["choices"]) # mmlu

    # generate answers
    answers = []
    for _ in tqdm(range(NUM_SAMPLE)):
        answer = llm.infer(prompt)
        answers.append(answer)

    # ground truth TODO
    # _gt = data[i]["answer"] # gsm8k, aime25, amc23
    # _gt = [data[i]["options"], data[i]["correct"]] # math_qa
    _gt = [data[i]["choices"], data[i]["answer"]] # mmlu
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
    with open(f"/data/yoshie/mtrths_labo/output_pass8_llama3_mmlu.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")