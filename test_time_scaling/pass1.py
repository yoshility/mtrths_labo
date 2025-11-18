import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama3:
    def __init__(self):
        # model and tokenizer
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="cuda")
        self.model.eval()

    def release_memory(self):
        del self.model, self.tokenizer

    def infer(self, prompt):
        with torch.no_grad():
            messages = [
                {"role": "system", "content": "You are a helpful assistant solving math problems."},
                {"role": "user", "content": prompt}
            ]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            input_tokens = self.tokenizer(input_text, return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **input_tokens, # set attention_mask by this
                # temperature=0.7, # random even if no temp
                max_new_tokens=1024,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            input_len = input_tokens.input_ids.shape[-1]
            gen_sequences = outputs.sequences[:, input_len:].cpu()
            decoded_text = self.tokenizer.decode(gen_sequences[0], skip_special_tokens=True)
            
            return decoded_text

def check_is_correct(gt, answer):
    answer_splited = answer.split("\n")
    # print(f"answer_splited:\n{answer_splited}")
    return int(gt in answer_splited[-1])

if __name__ == '__main__':
    model = Llama3()
    data = load_dataset("MathArena/aime_2025", split="train")
    for i in tqdm(range(len(data))):
        prompt = data[i]["problem"]

        # model output
        answer = model.infer(prompt)
        answer = model.infer(prompt, temperature=0.7)
        
        # ground truth
        gt = str(data[i]["answer"])

        # check the answer
        is_correct = check_is_correct(gt, answer)

        # save model outputs to file
        result = {
            "index": data[i]["problem_idx"],
            "question": prompt,
            "pred": answer,
            "answer": gt,
            "is_correct": is_correct
        }
        with open(f"/data/yoshie/mtrths_labo/output_pass@1_llama3_aime2025.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
