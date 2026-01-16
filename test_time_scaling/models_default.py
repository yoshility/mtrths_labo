import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Llama3:
    def __init__(self):
        # model and tokenizer
        print("Loading LLM (Llama3)")
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="cuda") # cuda
        self.model.eval()

    def release_memory(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()

    def infer(self, prompt):
        with torch.no_grad():
            messages = [
                {"role": "system", "content": "You are a helpful assistant solving math problems. Solve the problem step by step. Separate the steps with '\\n\\n' to make it readable."},
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
            
            return decoded_text, len(outputs.sequences[0])

class Qwen:
    def __init__(self):
        # model and tokenizer
        self.model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, device_map="cuda")
        self.model.eval()
    
    def release_memory(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
    
    def infer(self, prompt):
        with torch.no_grad():
            messages = [
                {"role": "system", "content": "You are a helpful assistant solving math problems. Solve the problem step by step. Separate the steps with '\\n\\n' to make it readable."},
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
                # do_sample=True,
                # temperature=0.8, # random even if no temp
                # top_p=0.9,
                # top_k=5, # ここ細かく調節しないとすぐハルシネーションする
                max_new_tokens=1024,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            input_len = input_tokens.input_ids.shape[-1]
            gen_sequences = outputs.sequences[:, input_len:].cpu()
            decoded_text = self.tokenizer.decode(gen_sequences[0], skip_special_tokens=True)
            
            return decoded_text, len(outputs.sequences[0])