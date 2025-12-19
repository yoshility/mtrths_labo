import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env を読み込む

FORGE_API_KEY = os.getenv("FORGE_API_KEY")
 
client = OpenAI(
    base_url="https://api.forge.tensorblock.co/v1", 
    api_key=FORGE_API_KEY,  
)

# LLM=Llama3.1-8B-Inst, PRM=Qwen2.5-Math-7B-PRM800K, Dataset=GSM8K
input_file = "/data/yoshie/mtrths_labo/output_BoNLast_llama3_gsm8k.jsonl"

# prompt base
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # load data
        data = json.loads(line)

        # make prompt content
        content = "# Instruction\nHere is a pair of question and answer. Please check whether each step in the answer is correct. If the step is correct, output 1, if otherwise, output 0. Finally, please only output a string of 0 and 1 with space separated. For example, when the step correctness of an answer is [correct, correct, wrong], then output \"1 1 0\".\n\n"
        content += f"# Question\n{data['question']}\n\n"
        splited_answer = data['pred'].split('\n\n')
        processed_answer = '# Answer\n'
        for i in range(len(splited_answer)):
            processed_answer += f"Step{i+1}: {splited_answer[i]}\n\n"
        content += processed_answer
        print(f"\ncontent\n----------------------\n{content}\n")

        # ask gpt
        completion = client.chat.completions.create(
            model="tensorblock/gpt-4o",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        print(f"gpt's answer: {completion.choices[0].message.content}\n")
        
        cnt += 1
        if cnt == 3:
            exit(0)

# Calculate point-biserial correlation
'''
r = (m1 - m0)\sqrt(p(1-p)) / s_x
'''
rpb_list = []
