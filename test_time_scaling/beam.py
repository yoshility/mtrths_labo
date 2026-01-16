# max_maxとほぼ同じことする
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

from models_step import Llama3, Qwen
from prm import Qwen800, QwenPRM
from utils import get_answer, check_is_correct

NUM_SAMPLE = 6
BEAM_SIZE = 4

# Total generated tokens
gen_token = 0

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--llm') # ['llama', 'qwen']
parser.add_argument('--prm') # ['qwen800', 'qwenPRM']
parser.add_argument('--data') # ['gsm8k', 'mmlu']
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
args = parser.parse_args()

# LLM
if args.llm == 'llama':
    llm = Llama3()
    eot_id = 128009 # <|eot_id|>==eos_token_id
elif args.llm == 'qwen':
    llm = Qwen()
    eot_id = 151645 # <|im_end|>==eos_token_id

# PRM
if args.prm == 'qwen800':
    prm = Qwen800()
elif args.prm == 'qwenPRM':
    prm = QwenPRM()

def beam_search(now_steps: list, num_sample: int, beam_size=4):
    # Generate <NUM_SAMPLE> steps
    children = [] # 4 * 6 samples basically
    tokenized_children = []

    # # LLM
    # if args.llm == 'llama':
    #     llm = Llama3()
    # elif args.llm == 'qwen':
    #     llm = Qwen()

    # llm = Llama3()
    for i in tqdm(range(len(now_steps)), desc="Generate next steps"):
        for _ in range(num_sample):
            next_step, tokenized_next_step = llm.generate_one_step(now_steps[i])
            children.append(next_step)
            tokenized_children.append(tokenized_next_step)
            global gen_token
            gen_token += len(tokenized_next_step[0]) # トークン数集計
    # llm.release_memory()

    # Assign PRM score
    # prm = PRM()
    for i in tqdm(range(len(children)), desc="Assign PRM scores"):
        score = prm.prm(children[i], return_all=False)
        children[i] = {"step": children[i], "score": score}
        tokenized_children[i] = {"step": tokenized_children[i], "score": score}
    # prm.release_memory()

    # debug
    # for i in range(len(children)):
    #     print(f"\nchildren[{i}]: ----------------------------------------------------------")
    #     print(children[i]['step'])
    #     print(f"score[{i}]: {children[i]['score']}")

    # Find the best <beam_size> children
    best_children = sorted(children, key=lambda x: x["score"], reverse=True)[:beam_size]
    best_tokenized_children = sorted(tokenized_children, key=lambda x: x["score"], reverse=True)[:beam_size]

    # debug
    # print("chosen children: --------------------------------------------------------------")
    # for i in range(len(best_children)):
    #     print(f"\nchosen children's score[{i}]:")
    #     print(best_children[i]['score'])
    
    return best_children, best_tokenized_children

def answer_question(index, messages):
    # Now step number
    num_step = 0

    now_steps = [messages] # beam_size = 4

    completed_steps = [] # steps with eot

    # Repeat until the generation stops
    while True:
        num_step += 1
        # Generate <NUM_SAMPLE> steps and return the best <BEAM_SIZE> steps with thier scores (dict)
        next_steps, tokenized_next_steps = beam_search(
            now_steps=now_steps,
            num_sample=NUM_SAMPLE,
            beam_size=BEAM_SIZE
        )
        # Only put the unfinished steps into now_steps list
        now_steps = []
        for i in range(len(next_steps)):
            if tokenized_next_steps[i]["step"][0][-1] == eot_id: # <eot_id>
                completed_steps.append(next_steps[i])
            elif num_step >= 20: # If too long, treat as completed and stop later (じゃないとcompleted_stepsが空になってしまう)
                completed_steps.append(next_steps[i])
            else:
                now_steps.append(next_steps[i]["step"])

        # debug
        print(f"\nQuestion index: {index}") # 現在の問題番号
        print("------------------------------")
        print(f"Now num_step: {num_step}") # 現在のステップ数
        print(f"len(completed_steps): {len(completed_steps)}") # 完成した回答の個数
        print(f"len(now_steps): {len(now_steps)}") # まだ生成中の回答の個数

        # No more now_steps, then finish
        if len(now_steps)==0:
            break       
        if num_step >= 20: # If too long, stop generation
            break
    
    # Choose the best step among completed_steps
    final_answer = max(completed_steps, key=lambda x: x["score"])
    return final_answer["step"]

if __name__ == '__main__':

    # dataset
    if args.data == 'gsm8k':
        data = load_dataset("openai/gsm8k", "main", split="train")
        dataset_name = 'gsm8k'
    elif args.data == 'mmlu':
        _data = load_dataset("TIGER-Lab/MMLU-STEM", split="test")
        data = [d for d in _data if d.get("subject") == "high_school_mathematics"]
        dataset_name = 'mmlu'

    for i in tqdm(range(args.start, args.end)):
        # generate answer
        if args.data == 'gsm8k':
            prompt = data[i]["question"] # gsm8k
        elif args.data == 'mmlu':
            prompt = "Please choose the correct answer of the following question from the options.\n# Question:\n" + data[i]["question"] + "\n# Options:\n" + str(data[i]["choices"]) # mmlu

        messages = [
            {"role": "system", "content": "You are a helpful assistant solving math problems. Solve the problem step by step. Separate the steps with '\\n\\n' to make it readable."},
            {"role": "user", "content": prompt}
        ]
        final_answer = answer_question(i, messages)
        
        # ground truth
        if args.data == 'gsm8k':
            _gt = data[i]["answer"] # gsm8k
        elif args.data == 'mmlu':
            _gt = [data[i]["choices"], data[i]["answer"]] # mmlu
        gt = get_answer(dataset_name, _gt)

        # check the answer
        is_correct = check_is_correct(dataset_name, gt, final_answer[2]["content"])

        # save model outputs to file
        result = {
            "index": i,
            "question": prompt,
            "pred": final_answer[2]["content"],
            "answer": _gt,
            "is_correct": is_correct
        }
        with open(f"/data/yoshie/mtrths_labo/output_beam_{args.llm}_{args.prm}_{args.data}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Now gen_token: {gen_token}")
        
    print(f"Total generated tokens(id:{args.start}~id:{args.end-1}): {gen_token}")
    print(f"Average generated tokens(id:{args.start}~id:{args.end-1}): {gen_token}/({args.end}-{args.start}) = {gen_token/(args.end-args.start)}")
