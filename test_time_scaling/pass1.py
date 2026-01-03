import json
from tqdm import tqdm
from datasets import load_dataset

from models_default import Llama3, Qwen
from utils import get_answer, check_is_correct

# TODO
dataset_name = 'gsm8k' # ['gsm8k', 'aime25', 'math_qa', 'amc23']

# TODO
# llm = Llama3()
llm = Qwen()

# TODO: dataset
data = load_dataset("openai/gsm8k", "main", split="train")
# data = load_dataset("MathArena/aime_2025", split="train")
# data = load_dataset("allenai/math_qa", split="train", trust_remote_code=True)
# data = load_dataset("math-ai/amc23", split="test")

for i in tqdm(range(2, 100)): # TODO
    # TODO
    prompt = data[i]["question"] # gsm8k, amc23
    # prompt = data[i]["problem"] # aime25
    # prompt = "Please choose the correct answer of the following question from the options.\n# Question:\n" + data[i]["Problem"] + "\n# Options:\n" + data[i]["options"] # math_qa

    # generate answer
    answer = llm.infer(prompt)

    # ground truth TODO
    _gt = data[i]["answer"] # gsm8k, aime25, amc23
    # _gt = [data[i]["options"], data[i]["correct"]] # math_qa
    gt = get_answer(dataset_name, _gt)

    # check if correct
    is_correct = check_is_correct(dataset_name, gt, answer)

    # save model outputs to file
    result = {
        "index": i,
        "question": prompt,
        "pred": answer,
        "answer": _gt,
        "is_correct": is_correct
    }
    with open(f"/data/yoshie/mtrths_labo/output_pass1_qwen_gsm8k.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")