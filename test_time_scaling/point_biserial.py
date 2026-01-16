import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--llm') # ['llama', 'qwen']
parser.add_argument('--prm') # ['qwen800', 'qwenPRM']
parser.add_argument('--data') # ['gsm8k', 'mmlu']
args = parser.parse_args()

# setting
num_step_range = 10 # default: 10 区間の数
interval = 100 // num_step_range # 各区間の幅
# file
input_file = f"/data/yoshie/mtrths_labo/pre_rpb_{args.llm}_{args.prm}_{args.data}_ver2.csv" # gptでラベル付けしたもの
output_file = f"/data/yoshie/mtrths_labo/rpb_{args.llm}_{args.prm}_{args.data}_ver2.csv" # step_pct順に並べたもの
output_fig = f"/data/yoshie/mtrths_labo/rpb_{args.llm}_{args.prm}_{args.data}_ver2.png" # step_pctごとにrpbをプロットしたもの
output_fig_m1m0 = f"/data/yoshie/mtrths_labo/rpb_m1m0_{args.llm}_{args.prm}_{args.data}_ver2.png" # step_pctごとにM1,M0,Sx,pをプロットしたもの

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
m1m0_list = [[None] * num_step_range for _ in range(4)] # m1, m0, Sx, n1/n
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
    m1m0_list[0][i] = m1
    m1m0_list[1][i] = m0
    m1m0_list[2][i] = sigma_x
    m1m0_list[3][i] = n1/n
    print(f"m1: {m1}\nm0: {m0}\nn1: {n1}\nn0: {n0}\nsigma_x: {sigma_x}\nrpb: {rpb_list[i]}")

# Plot point-biserial correlation with num_step
print(f"\nstep_list: {step_list}")
print(f"rpb_list: {rpb_list}")
if args.data == 'gsm8k':
    color, label = 'blue', 'GSM8K'
elif args.data == 'mmlu':
    color, label = 'red', 'MMLU'
fig = plt.figure(1)
plt.plot(step_list, rpb_list, color=color, label=label, marker='o')
plt.title(f'Point-biserial correlation ({args.llm}_{args.prm}_{args.data})')
plt.xlabel('step (%)')
plt.ylabel('correlation')
plt.xlim(-10, 100)
plt.ylim(-0.1, 1)
plt.legend()
plt.savefig(output_fig)

# plot m1, m0, sigma_x, n1/n with num_step
fig = plt.figure(2)
plt.plot(step_list, m1m0_list[0], label='M1', marker='o') # m1
plt.plot(step_list, m1m0_list[1], label='M0', marker='o') # m0
plt.plot(step_list, m1m0_list[2], label='σ_x', marker='o') # Sx
plt.plot(step_list, m1m0_list[3], label='n1/n', marker='o') # n1/n
plt.title(f"Components of rpb ({args.llm}_{args.prm}_{args.data})")
plt.xlabel('step (%)')
plt.ylabel('components')
plt.xlim(-10, 100)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
fig.tight_layout() 
plt.savefig(output_fig_m1m0)
