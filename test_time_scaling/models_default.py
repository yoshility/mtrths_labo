import torch
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
                max_new_tokens=2048,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            input_len = input_tokens.input_ids.shape[-1]
            gen_sequences = outputs.sequences[:, input_len:].cpu()
            decoded_text = self.tokenizer.decode(gen_sequences[0], skip_special_tokens=True)
            
            return decoded_text