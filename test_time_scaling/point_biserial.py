import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# LLM=Llama3.1-8B-Inst, PRM=Qwen2.5-Math-7B-PRM800K, Dataset=GSM8K
input_file = "/data/yoshie/mtrths_labo/pre_rpb_llama3_qwen800_gsm8k.csv"
output_file = "/data/yoshie/mtrths_labo/rpb_llama3_qwen800_gsm8k.csv"
output_fig = "/data/yoshie/mtrths_labo/rpb_llama3_qwen800_gam8k.png"

# Calculate point-biserial correlation
'''
r = (m1 - m0)\sqrt(n1*n0/n^2) / s_x
'''
df = pd.read_csv(input_file)
step_list = np.arange(10, 110, 10) # percent
df['step_pct'] = df['num_step'] / df['all_step'] * 100
rpb_list = [None] * 10
for i in range(10):
    df_tmp = df[(i*10<df['step_pct']) & (df['step_pct']<=(i+1)*10)]
    print(f"\n{i*10}<step_pct<={(i+1)*10} data size: {len(df_tmp)}")
    m1 = np.mean(df_tmp[df_tmp['label']==1]['prm_score'])
    m0 = np.mean(df_tmp[df_tmp['label']==0]['prm_score'])
    n1 = len(df_tmp[df_tmp['label']==1])
    n0 = len(df_tmp[df_tmp['label']==0])
    n = len(df_tmp)
    sigma_x = np.std(df_tmp['prm_score'])
    rpb = (m1-m0) * np.sqrt(n1*n0/n/n) / sigma_x
    # if rpb == nan, set rpb = 0 (相関が0というわけではなく、データがそもそも集まらなかったということ)
    rpb_list[i] = 0 if np.isnan(rpb) else rpb
    print(f"m1: {m1}\nm0: {m0}\nn1: {n1}\nsigma_x: {sigma_x}\nrpb: {rpb_list[i]}")

# Plot point-biserial correlation with num_step
print(f"step_list: {step_list}")
print(f"rpb_list: {rpb_list}")
plt.plot(step_list, rpb_list, 'o-')
plt.title('Point-biserial correlation (llama3_qwen800_gsm8k)')
plt.xlabel('step (%)')
plt.ylabel('correlation')
plt.savefig(output_fig)
