import os
import csv
import json
from tqdm import tqdm
from itertools import islice
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # .env を読み込む

FORGE_API_KEY = os.getenv("FORGE_API_KEY")
 
client = OpenAI(
    base_url="https://api.forge.tensorblock.co/v1", 
    api_key=FORGE_API_KEY,  
)

# LLM=Llama3.1-8B-Inst, PRM=Qwen2.5-Math-7B-PRM800K, Dataset=GSM8K
input_file = "/data/yoshie/mtrths_labo/output_BoNLast_llama3_qwen800_gsm8k_allCandidates.jsonl"
output_file = "/data/yoshie/mtrths_labo/pre_rpb_llama3_qwen800_gsm8k.csv"

# make output file
if not os.path.exists(output_file):
    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['question_index', 'candidate_index', 'num_step', 'all_step', 'prm_score', 'label'])

with open(input_file, 'r', encoding='utf-8') as f:
    for line in tqdm(islice(f, 180, None)): # set end=None if all
        # load model output data
        data = json.loads(line)

        # make prompt content
        splited_answer = data['pred'].split('\n\n')
        content = f"# Instruction\nHere is a pair of question and answer. Please check whether each step in the answer is correct. If the step is correct for the question, output 1, if it is misleading or unuseful for the question, output 0. Finally, please only output a string of 0 and 1 with space separated. For example, when the step correctness of an answer is [correct, correct, wrong], then output 1 1 0. Don't output any quotation marks or periods. Make sure the length of your output is equal to the number of the steps: {len(splited_answer)}!!! You don't have to care about the step numbers which are not written as \"Step:~\".\n\n"
        content += f"# Question\n{data['question']}\n\n"
        processed_answer = '# Answer\n'
        for i in range(len(splited_answer)):
            processed_answer += f"Step{i+1}: {splited_answer[i]}\n\n"
        content += processed_answer

        # ask gpt to get labels
        completion = client.chat.completions.create(
            model="tensorblock/gpt-4o",
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
        )
        labels = completion.choices[0].message.content
        labels = [int(l) for l in labels.rstrip('.').split()]
        if len(data['scores']) != len(labels):
            labels = [1] * len(data['scores']) # 一旦ごまかす
            print(f"[Warning] len(data['scores'])!=len(labels) | index: {data['index']} | candidate: {data['candidate']}")

        # make prm scores and labels data
        csv_data = []
        for i in range(len(labels)):
            # [index, candidate_index, num_step, all_step, prm_score, label]
            csv_data.append([data['index'], data['candidate'], i+1, len(labels), data['scores'][i], labels[i]])

        # save prm scores and labels into csv
        with open(output_file, 'a') as fo:
            writer = csv.writer(fo)
            writer.writerows(csv_data)
