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
dataset1 = 'gsm8k'
dataset2 = 'mmlu'
num_step_range = 10 # default: 10 区間の数
interval = 100 // num_step_range # 各区間の幅
# file
input_file1 = f"/data/yoshie/mtrths_labo/pre_rpb_{args.llm}_{args.prm}_{dataset1}_ver2.csv" # gptでラベル付けしたもの
input_file2 = f"/data/yoshie/mtrths_labo/pre_rpb_{args.llm}_{args.prm}_{dataset2}_ver2.csv" # gptでラベル付けしたもの
output_file = f"/data/yoshie/mtrths_labo/rpb_{args.llm}_{args.prm}_{dataset1}_{dataset2}_ver2.csv" # step_pct順に並べたもの
output_fig = f"/data/yoshie/mtrths_labo/rpb_{args.llm}_{args.prm}_{dataset1}_{dataset2}_ver2.png" # step_pctごとにrpbをプロットしたもの

# Calculate point-biserial correlation
'''
r = (m1 - m0)\sqrt(n1*n0/n^2) / s_x
'''
df1 = pd.read_csv(input_file1)
df2 = pd.read_csv(input_file2)
step_list = np.arange(0, 100, 100//num_step_range) # graph's x-axis (percent)
df1['step_pct'] = (df1['num_step']-1) / (df1['all_step']-1) * 100
df2['step_pct'] = (df2['num_step']-1) / (df2['all_step']-1) * 100
df1['dataset_name'] = 'gsm8k'
df2['dataset_name'] = 'mmlu'
df_all = pd.concat([df1, df2], ignore_index=True) # インデックスはつけなおす
df_pct = df_all.sort_values('step_pct')
df_pct.to_csv(output_file) # statistic data（結合したものだけを出力する）
rpb_list1 = [None] * num_step_range # graph's y-axis (gsm8k)
rpb_list2 = [None] * num_step_range # graph's y-axis (mmlu)
rpb_list_all = [None] * num_step_range # graph's y-axis (gsm8k + mmlu)
for i in range(num_step_range):
    if i == num_step_range-1: # final step range
        df_tmp1 = df1[(i*interval<=df1['step_pct']) & (df1['step_pct']<=(i+1)*interval)]
        df_tmp2 = df2[(i*interval<=df2['step_pct']) & (df2['step_pct']<=(i+1)*interval)]
        df_tmp_all = df_all[(i*interval<=df_all['step_pct']) & (df_all['step_pct']<=(i+1)*interval)]
        print(f"\n{i*interval}<=step_pct<={(i+1)*interval} all data size: {len(df_tmp_all)}")
    else:
        df_tmp1 = df1[(i*interval<=df1['step_pct']) & (df1['step_pct']<(i+1)*interval)]
        df_tmp2 = df2[(i*interval<=df2['step_pct']) & (df2['step_pct']<(i+1)*interval)]
        df_tmp_all = df_all[(i*interval<=df_all['step_pct']) & (df_all['step_pct']<(i+1)*interval)]
        print(f"\n{i*interval}<=step_pct<{(i+1)*interval} all data size: {len(df_tmp_all)}")
    
    # df_tmp1
    m1 = np.mean(df_tmp1[df_tmp1['label']==1]['prm_score'])
    m0 = np.mean(df_tmp1[df_tmp1['label']==0]['prm_score'])
    n1 = len(df_tmp1[df_tmp1['label']==1])
    n0 = len(df_tmp1[df_tmp1['label']==0])
    n = len(df_tmp1)
    sigma_x = np.std(df_tmp1['prm_score'])
    rpb = (m1-m0) * np.sqrt(n1*n0/n/n) / sigma_x
    rpb_list1[i] = rpb # rpb = nanのときは、そもそもデータ集まらなかったということ

    # df_tmp2
    m1 = np.mean(df_tmp2[df_tmp2['label']==1]['prm_score'])
    m0 = np.mean(df_tmp2[df_tmp2['label']==0]['prm_score'])
    n1 = len(df_tmp2[df_tmp2['label']==1])
    n0 = len(df_tmp2[df_tmp2['label']==0])
    n = len(df_tmp2)
    sigma_x = np.std(df_tmp2['prm_score'])
    rpb = (m1-m0) * np.sqrt(n1*n0/n/n) / sigma_x
    rpb_list2[i] = rpb
    
    # df_tmp_all
    m1 = np.mean(df_tmp_all[df_tmp_all['label']==1]['prm_score'])
    m0 = np.mean(df_tmp_all[df_tmp_all['label']==0]['prm_score'])
    n1 = len(df_tmp_all[df_tmp_all['label']==1])
    n0 = len(df_tmp_all[df_tmp_all['label']==0])
    n = len(df_tmp_all)
    sigma_x = np.std(df_tmp_all['prm_score'])
    rpb = (m1-m0) * np.sqrt(n1*n0/n/n) / sigma_x
    rpb_list_all[i] = rpb
    print(f"m1: {m1}\nm0: {m0}\nn1: {n1}\nn0: {n0}\nsigma_x: {sigma_x}\nrpb: {rpb_list_all[i]}")

# Plot point-biserial correlation with num_step
print(f"\nstep_list: {step_list}")
print(f"rpb_list: {rpb_list_all}")
plt.plot(step_list, rpb_list1, label='GSM8K', color='blue', marker='o') # gsm8k
plt.plot(step_list, rpb_list2, label='MMLU', color='red', marker='o') # mmlu
plt.plot(step_list, rpb_list_all, label='All data', color='green', marker='o') # all
plt.title(f'Point-biserial correlation ({args.llm}_{args.prm})')
plt.xlabel('step (%)')
plt.ylabel('correlation')
plt.xlim(-10, 100)
plt.ylim(-0.1, 1)
plt.legend()
plt.savefig(output_fig)
