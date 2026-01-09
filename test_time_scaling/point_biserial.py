import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# setting
llm = 'llama' # llama, qwen
prm = 'qwenPRM' # qwen800, qwenPRM
dataset = 'mmlu' # gsm8k, mmlu
num_step_range = 10 # default: 10 区間の数
interval = 100 // num_step_range # 各区間の幅
# file
input_file = f"/data/yoshie/mtrths_labo/pre_rpb_{llm}_{prm}_{dataset}.csv" # gptでラベル付けしたもの
output_file = f"/data/yoshie/mtrths_labo/rpb_{llm}_{prm}_{dataset}.csv" # step_pct順に並べたもの
output_fig = f"/data/yoshie/mtrths_labo/rpb_{llm}_{prm}_{dataset}.png" # step_pctごとにrpbをプロットしたもの

# Calculate point-biserial correlation
'''
r = (m1 - m0)\sqrt(n1*n0/n^2) / s_x
'''
df = pd.read_csv(input_file)
step_list = np.arange(0, 100, 100//num_step_range) # graph's x-axis (percent)
df['step_pct'] = (df['num_step']-1) / (df['all_step']-1) * 100
df_pct = df.sort_values('step_pct')
df_pct.to_csv(output_file) # statistic data
rpb_list = [None] * num_step_range # graph's y-axis
for i in range(num_step_range):
    if i == num_step_range-1: # final step range
        df_tmp = df[(i*interval<=df['step_pct']) & (df['step_pct']<=(i+1)*interval)]
        print(f"\n{i*interval}<=step_pct<={(i+1)*interval} data size: {len(df_tmp)}")
    else:
        df_tmp = df[(i*interval<=df['step_pct']) & (df['step_pct']<(i+1)*interval)]
        print(f"\n{i*interval}<=step_pct<{(i+1)*interval} data size: {len(df_tmp)}")
    m1 = np.mean(df_tmp[df_tmp['label']==1]['prm_score'])
    m0 = np.mean(df_tmp[df_tmp['label']==0]['prm_score'])
    n1 = len(df_tmp[df_tmp['label']==1])
    n0 = len(df_tmp[df_tmp['label']==0])
    n = len(df_tmp)
    sigma_x = np.std(df_tmp['prm_score'])
    rpb = (m1-m0) * np.sqrt(n1*n0/n/n) / sigma_x
    # rpb = nanのときは、そもそもデータ集まらなかったということ
    rpb_list[i] = rpb
    print(f"m1: {m1}\nm0: {m0}\nn1: {n1}\nn0: {n0}\nsigma_x: {sigma_x}\nrpb: {rpb_list[i]}")

# Plot point-biserial correlation with num_step
print(f"\nstep_list: {step_list}")
print(f"rpb_list: {rpb_list}")
plt.plot(step_list, rpb_list, 'o-')
plt.title(f'Point-biserial correlation ({llm}_{prm}_{dataset})')
plt.xlabel('step (%)')
plt.ylabel('correlation')
plt.xlim(0, 110)
plt.ylim(0, 1)
plt.savefig(output_fig)
