import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from prm import PRM

# model and tokenizer
class Llama3:
    def __init__(self):
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
        del self.tokenizer, self.model

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

def static_value(step) -> float:
    print(f"\n[Loading PRM ...]\n")
    prm = PRM()
    print(f"\n[PRM loaded]\n")
    step_reward = prm.prm(step, return_all=False)
    prm.release_memory()
    print(f"\n[PRM memory released]\n")
    return step_reward

# min-max -> max-max
def max_max(step: list, child_id: int, depth: int, child_num: int):
    # print(f"\n[Start max-max (depth={depth})]\n")
    # 読み深さに達した
    if depth == 0:
        # print("読み深さに達した")
        return static_value(step), None, None
    # 子ノード生成
    children = []
    tokenized_children = []
    print(f"\n[Loading llm ...]\n")
    llm = Llama3()
    print(f"\n[llm loaded]\n")
    for i in range(child_num):
        # print(f"\n[generate children[{child_id+str(i)}]]\n")
        next_step, tokenized_next_step = llm.generate_one_step(step)
        children.append(next_step)
        tokenized_children.append(tokenized_next_step)
        print(f"\nchildren[{child_id+str(i)}] generated:\n{children[i]}\n")
    llm.release_memory()
    print(f"\n[llm memory released]\n")
    
    max_score = -1e4
    max_index = 0
    for i in range(child_num):
        # print(f"\n[max_max(children[{child_id+str(i)}])]\n")
        score, _, _ = max_max(children[i], child_id+str(i), depth-1, child_num)
        # print(f"\nscore[{child_id+str(i)}]: {score}\n")
        if score > max_score:
            max_score = score
            max_index = i
    # print(f"\nChoose children[{max_index}] (score: {max_score})\n")
    
    return max_score, children[max_index], tokenized_children[max_index]

# 繰り返し次のステップを決めながら回答を生成していく
def answer_question(messages, return_all=False):
    # 現在のステップ数
    num_step = 0

    now_step = messages

    # 生成が終わるまでmax-maxを繰り返す
    while True:
        num_step += 1
        print(f"\n<<num_step: {num_step}>>\n")
        _, next_step, tokenized_next_step = max_max(
            step=now_step,
            child_id=f"{num_step}-C", # for debug, assign an id to each node
            depth=1,
            child_num=3
        )
        now_step = next_step
        # print(f"\nchosen_next_step:\n{next_step}\n")
        if num_step >= 20:
            return now_step[2]["content"] + "(too long, stop generation)"
        # もし選んだ次ステップでEOSなら、生成終了
        # if tokenized_next_step[0][-1] == llm.tokenizer.eos_token_id:
        if tokenized_next_step[0][-1] == 128009:
            print(f"\n生成すべて完了\n")
            break
        # input('Press Enter to continue ↲')
    
    if return_all:
        return now_step
    else:
        return now_step[2]["content"]

if __name__ == '__main__':
    # 初期化
    # Use chat template -> avoid hallucination
    # Input
    prompt = "Each bird eats 12 beetles per day, each snake eats 3 birds per day, and each jaguar eats 5 snakes per day. If there are 6 jaguars in a forest, how many beetles are eaten each day?"
    messages = [
        {"role": "system", "content": "You are a helpful assistant solving math problems."},
        {"role": "user", "content": prompt}
    ]
    answer = answer_question(messages, return_all=False)
    print(f"\nfinal answer:\n{answer}\n")    