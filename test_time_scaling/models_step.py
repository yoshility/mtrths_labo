import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

class Llama3:
    def __init__(self):
        print("Loading Llama3")
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="cuda", # if auto, offload to cpu -> slow
            offload_buffers=True
        )
        self.model.eval()

    def release_memory(self):
        print("Release Llama3")
        del self.tokenizer, self.model
        torch.cuda.empty_cache()

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., -1, None]] = -float('Inf')
        return out

    def generate_one_step(self, step):
        # stepはsystem,user(,assistant)を含む辞書のリスト
        # これに対してダブル改行が出るまで続きを生成する

        # If system & user, add <|eot_id|><|start_header_id|>assistant<|end_header_id|>.
        # If system & user & assistant, add <|eot_id|> only. (But change it to ' \n\n' later.)
        input_text = self.tokenizer.apply_chat_template(
            step,
            tokenize=False,
            add_generation_prompt=(len(step)==2)
        )
        # print(f"\ninput_text:\n{[input_text]}\n")
        input_tokens = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        generated_ids = input_tokens["input_ids"]
        # print(f"\ninput_ids(before):\n{generated_ids}\n")
        # If the last token is '<|eot_id|>'(128009), change it to ' \n\n'(4815)
        if generated_ids[0, -1].item() == self.tokenizer.eos_token_id:
            generated_ids = generated_ids[:, :-1]
            generated_ids = torch.cat([generated_ids, torch.tensor([[4815]], device='cuda:0')], dim=-1)
        # print(f"\ninput_ids(after):\n{generated_ids}\n")

        past_key_values = DynamicCache()

        with torch.no_grad():
            for t in range(200): # max token
                if t == 0:
                    input_ids = generated_ids
                else:
                    input_ids = generated_ids[:, -1:]
                outputs = self.model.forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                temperature = 0.8
            
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                
                # choose the next token
                adjusted_logits = logits[:, -1, :] / temperature
                topk_logits = self.top_k_logits(adjusted_logits, k=20)
                probs = torch.softmax(topk_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1) # random sampling based on probs
                
                # add to generated tokens
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                # decode to text (ちゃんとダブル改行検知できてるかチェック)
                # decoded_next_token_id = tokenizer.decode(next_token_id[0])
                # enter_exist = '\n\n' in decoded_next_token_id
                # print(f"[{decoded_next_token_id}]:[{enter_exist}]", end=" ")

                del outputs, logits, topk_logits, probs

                # if \n\n or EOS, stop generation
                if "\n\n" in self.tokenizer.decode(next_token_id[0]):
                    # print("\n- \\n\\n detected. Stopping generation.")
                    break
                elif next_token_id.item() == self.tokenizer.eos_token_id:
                    # print("\n- EOS detected. Stopping generation.")
                    break
        
        generated_ids = generated_ids.detach().cpu()
        generated_text = ''.join([self.tokenizer.decode(w, skip_special_tokens=True) for w in generated_ids[0]][(input_tokens.input_ids.shape[-1]):])
        # print(f"\ngenerated_text(all):\n{tokenizer.decode(generated_ids[0].tolist())}\n")
        next_step = copy.deepcopy(step)
        if len(step) == 2: # systemとuserのみのとき
            next_step.append(
                {"role": "assistant", "content": generated_text}
            )
        else: # assistantすでにあるとき
            next_step[2]["content"] += generated_text
        return next_step, generated_ids

class Qwen:
    def __init__(self):
        print("Loading Qwen")
        self.model_id = "Qwen/Qwen2.5-Math-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="cuda", # if auto, offload to cpu -> slow
            offload_buffers=True
        )
        self.model.eval()
    
    def release_memory(self):
        print("Release Qwen")
        del self.tokenizer, self.model
        torch.cuda.empty_cache()
    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., -1, None]] = -float('Inf')
        return out

    def generate_one_step(self, step):
        # stepはsystem,user(,assistant)を含む辞書のリスト
        # これに対してダブル改行が出るまで続きを生成する

        # If system & user, add <|im_end|>\n<|im_start|>assistant\n
        # If system & user & assistant, add <|im_end|>\n only. (But delete them later.)
        input_text = self.tokenizer.apply_chat_template(
            step,
            tokenize=False,
            add_generation_prompt=(len(step)==2)
        )
        # print(f"\ninput_text:\n{[input_text]}\n")
        input_tokens = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        generated_ids = input_tokens["input_ids"]
        # print(f"\ninput_ids(before):\n{generated_ids}\n")
        # If the last tokens are '<|im_end|>'(151645) + '\n'(198), delete them
        if generated_ids[0, -2].item() == self.tokenizer.eos_token_id: # == <|im_end|>(151645)
            generated_ids = generated_ids[:, :-2] # なにもつけない
            # generated_ids = torch.cat([generated_ids, torch.tensor([[4815]], device='cuda:0')], dim=-1)
        # print(f"\ninput_ids(after):\n{generated_ids}\n")

        past_key_values = DynamicCache()

        with torch.no_grad():
            for t in range(200): # max token
                if t == 0:
                    input_ids = generated_ids
                else:
                    input_ids = generated_ids[:, -1:]
                outputs = self.model.forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                temperature = 1.0 # もともとは0.8
            
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                
                # choose the next token
                adjusted_logits = logits[:, -1, :] / temperature
                topk_logits = self.top_k_logits(adjusted_logits, k=5) # もともとはk=20
                probs = torch.softmax(topk_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1) # random sampling based on probs
                
                # add to generated tokens
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                # decode to text (ちゃんとダブル改行検知できてるかチェック)
                # decoded_next_token_id = tokenizer.decode(next_token_id[0])
                # enter_exist = '\n\n' in decoded_next_token_id
                # print(f"[{decoded_next_token_id}]:[{enter_exist}]", end=" ")

                del outputs, logits, topk_logits, probs

                # if \n\n or EOS, stop generation
                if "\n\n" in self.tokenizer.decode(next_token_id[0]):
                    # print("\n- \\n\\n detected. Stopping generation.")
                    break
                elif next_token_id.item() == self.tokenizer.eos_token_id:
                    # print("\n- EOS detected. Stopping generation.")
                    break
        
        generated_ids = generated_ids.detach().cpu()
        generated_text = ''.join([self.tokenizer.decode(w, skip_special_tokens=True) for w in generated_ids[0]][(input_tokens.input_ids.shape[-1]):])
        # print(f"\ngenerated_text(all):\n{tokenizer.decode(generated_ids[0].tolist())}\n")
        next_step = copy.deepcopy(step)
        if len(step) == 2: # systemとuserのみのとき
            next_step.append(
                {"role": "assistant", "content": generated_text}
            )
        else: # assistantすでにあるとき
            next_step[2]["content"] += generated_text
        return next_step, generated_ids
