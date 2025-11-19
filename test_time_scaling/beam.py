# max_maxとほぼ同じことする
import json
from tqdm import tqdm
from datasets import load_dataset

from max_max import Llama3
from prm import PRM

NUM_SAMPLE = 6
BEAM_SIZE = 4

def get_answer(answer):
    i = -1
    ans = ""
    while (answer[i] != ' '):
        ans = answer[i] + ans
        i -= 1
    return ans

def check_is_correct(gt, pred):
    # put ',' in gt (e.g. 488000 -> 488,000)
    i = len(gt)-1
    cnt = 0
    new_gt = ""
    while i >= 0:
        if cnt != 0 and cnt % 3 == 0:
            new_gt = ',' + new_gt
        new_gt = gt[i] + new_gt
        cnt += 1
        i -= 1
    
    # get final sentence from answer
    new_pred = pred.split('\n\n')[-1]

    return int(new_gt in new_pred or gt in new_pred)

def beam_search(now_steps: list, num_sample: int, beam_size=4):
    # Generate <num_sampling> steps
    children = [] # 6 samples basically
    tokenized_children = []
    llm = Llama3()
    for i in tqdm(range(len(now_steps))):
        for _ in range(num_sample):
            next_step, tokenized_next_step = llm.generate_one_step(now_steps[i])
            children.append(next_step)
            tokenized_children.append(tokenized_next_step)
    llm.release_memory()

    # Assign PRM score
    prm = PRM()
    for i in tqdm(range(len(children))):
        score = prm.prm(children[i], return_all=False)
        children[i] = {"step": children[i], "score": score}
        tokenized_children[i] = {"step": tokenized_children[i], "score": score}
    prm.release_memory()

    # Find the best <beam_size> children
    best_children = sorted(children, key=lambda x: x["score"], reverse=True)[:beam_size]
    best_tokenized_children = sorted(tokenized_children, key=lambda x: x["score"], reverse=True)[:beam_size]
    
    return best_children, best_tokenized_children

def answer_question(messages):
    # Now step number
    num_step = 0

    now_steps = [messages] # beam_size = 4

    completed_steps = [] # steps with eot

    # Repeat until the generation stops
    while True:
        num_step += 1
        # Generate <num_sampling> steps and return the best <beam_size> steps with thier scores (dict)
        next_steps, tokenized_next_steps = beam_search(
            now_steps=now_steps,
            num_sample=NUM_SAMPLE,
            beam_size=BEAM_SIZE
        )
        # Only put the unfinished steps into now_steps list
        now_steps = []
        for i in range(len(next_steps)):
            if tokenized_next_steps[i]["step"][0][-1] == 128009:
                completed_steps.append(next_steps[i])
            else:
                now_steps.append(next_steps[i]["step"])
        # No more now_steps, then finish
        if len(now_steps)==0:
            break
        
        if num_step >= 20:
            break
    
    # Choose the best <beam_size> steps among completed_steps
    final_answer = max(completed_steps, key=lambda x: x["score"])
    return final_answer["step"]

if __name__ == '__main__':
    data = load_dataset("openai/gsm8k", "main", split="train")

    for i in tqdm(range(100)):
        # generate answer
        prompt = data[i]["question"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant solving math problems."},
            {"role": "user", "content": prompt}
        ]
        final_answer = answer_question(messages)
        
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
            "answer": _gt,
            "is_correct": is_correct
        }
        with open(f"/data/yoshie/mtrths_labo/output_beam_llama3_gsm8k.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
