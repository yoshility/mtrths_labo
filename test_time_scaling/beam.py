# max_maxとほぼ同じことする
import json
from tqdm import tqdm
from datasets import load_dataset

from max_max import Llama3
from prm import PRM
from utils import get_answer, check_is_correct

NUM_SAMPLE = 6
BEAM_SIZE = 4

def beam_search(now_steps: list, num_sample: int, beam_size=4):
    # Generate <NUM_SAMPLE> steps
    children = [] # 4 * 6 samples basically
    tokenized_children = []
    llm = Llama3()
    for i in tqdm(range(len(now_steps)), desc="Generate next steps"):
        for _ in range(num_sample):
            next_step, tokenized_next_step = llm.generate_one_step(now_steps[i])
            children.append(next_step)
            tokenized_children.append(tokenized_next_step)
    llm.release_memory()

    # Assign PRM score
    prm = PRM()
    for i in tqdm(range(len(children)), desc="Assign PRM scores"):
        score = prm.prm(children[i], return_all=False)
        children[i] = {"step": children[i], "score": score}
        tokenized_children[i] = {"step": tokenized_children[i], "score": score}
    prm.release_memory()

    # Find the best <beam_size> children
    best_children = sorted(children, key=lambda x: x["score"], reverse=True)[:beam_size]
    best_tokenized_children = sorted(tokenized_children, key=lambda x: x["score"], reverse=True)[:beam_size]
    
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
            if tokenized_next_steps[i]["step"][0][-1] == 128009: # <eot_id>
                completed_steps.append(next_steps[i])
            elif num_step >= 20: # If too long, treat as completed and stop later (じゃないとcompleted_stepsが空になってしまう)
                completed_steps.append(next_steps[i])
            else:
                now_steps.append(next_steps[i]["step"])
        # No more now_steps, then finish
        if len(now_steps)==0:
            break
        
        if num_step >= 20: # If too long, stop generation
            break

        print(f"Question index: {index}") # 現在の問題番号
        print(f"Now num_step: {num_step}") # 現在のステップ数
        print(f"len(completed_steps): {len(completed_steps)}") # 完成した回答の個数
        print(f"len(now_steps): {len(now_steps)}") # まだ生成中の回答の個数
    
    # Choose the best step among completed_steps
    final_answer = max(completed_steps, key=lambda x: x["score"])
    return final_answer["step"]

if __name__ == '__main__':
    # TODO
    dataset_name = 'aime25' # ['gsm8k', 'aime25']

    # TODO
    # data = load_dataset("openai/gsm8k", "main", split="train")
    data = load_dataset("MathArena/aime_2025", split="train")

    for i in tqdm(range(30)): # TODO
        # generate answer
        print(f"Question index: {i}\n-----------------------")
        # TODO
        # prompt = data[i]["question"] # gsm8k
        prompt = data[i]["problem"] # aime25
        messages = [
            {"role": "system", "content": "You are a helpful assistant solving math problems."},
            {"role": "user", "content": prompt}
        ]
        final_answer = answer_question(i, messages)
        
        # ground truth
        _gt = data[i]["answer"]
        gt = get_answer(dataset_name, _gt)

        # check the answer
        is_correct = check_is_correct(dataset_name, gt, final_answer[2]["content"])

        # save model outputs to file
        result = {
            "index": i,
            "question": prompt,
            "pred": final_answer,
            "answer": _gt,
            "is_correct": is_correct
        }
        with open(f"/data/yoshie/mtrths_labo/output_beam_llama3_aime25.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
