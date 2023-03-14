# ライブラリ
import numpy as np
import model
import statistics
from scipy.stats import entropy
from tqdm import tqdm

# 共通設定
K=0 # 仮説数K+1
alpha=0.9
m=1.6
H_likelihoods_1 = np.random.rand(K+1) # 尤度の初期値ランダム
T_likelihoods_1 = 1 - H_likelihoods_1 # 裏の尤度 C(d=0|h_max) (31)式

H_likelihoods_2 = np.random.rand(K+1) # 尤度の初期値ランダム
T_likelihoods_2 = 1 - H_likelihoods_2 # 裏の尤度 C(d=0|h_max) (31)式

prior_probs_1 = np.array([1/(K+1)]*len(H_likelihoods_1)) # 事前分布： 一様分布 (32)式
prior_probs_2 = np.array([1/(K+1)]*len(H_likelihoods_2)) # 事前分布： 一様分布 (32)式
d1= 1 if np.random.rand() < 0.5 else 0
d2= 1 if np.random.rand() < 0.5 else 0

agent1 = model.Agent(K, alpha, m)
agent2 = model.Agent(K, alpha, m)

SIM_NUM = 100
corr_list = np.zeros(SIM_NUM)
entropy_list = np.zeros(SIM_NUM)
mutual_list = np.zeros(SIM_NUM)
for sim_idx in tqdm(range(SIM_NUM)):
  ## シミュレーション ##
  TOSS_NUM=20000 # 試行回数
  timestep = np.arange(0,TOSS_NUM) # 横軸
  estimation_1 = np.zeros(TOSS_NUM)  # ベイズ推定量の推移
  estimation_2 = np.zeros(TOSS_NUM)  # ベイズ推定量の推移

  h = np.zeros((K+1, TOSS_NUM))
  H_h = np.zeros((K+1, TOSS_NUM))

  #### ベイズ更新 ####
  for i in range(TOSS_NUM):
    prior_probs_copy_1 = np.copy(prior_probs_1)
    C_d_1 = agent1.bayes_update(prior_probs_1, H_likelihoods_1, T_likelihoods_1, d2)
    H_likelihoods_1, T_likelihoods_1 = agent1.Inverse_bayes_update(prior_probs_copy_1, H_likelihoods_1, T_likelihoods_1, d2, C_d_1)
    estimation_1[i] = np.sum(prior_probs_1*H_likelihoods_1)  # (30)式  ベイズ推定
    d1 = agent1.generator(estimation_1[i]) # generator(表確率の確信度)

    prior_probs_copy_2 = np.copy(prior_probs_2)
    C_d_2 = agent2.bayes_update(prior_probs_2, H_likelihoods_2, T_likelihoods_2, d1)
    H_likelihoods_2, T_likelihoods_2 = agent2.Inverse_bayes_update(prior_probs_copy_2, H_likelihoods_2, T_likelihoods_2, d1, C_d_2)
    estimation_2[i] = np.sum(prior_probs_2*H_likelihoods_2)  # (30)式
    d2 = agent2.generator(estimation_2[i]) # generator(表確率の確信度)

    ### plot 用 ###
    for j in range(K+1):
      h[j][i] = prior_probs_1[j]
      H_h[j][i] = H_likelihoods_1[j]
    #############

  corr_list[sim_idx] = np.corrcoef(estimation_1, estimation_2)[0][1]
  entropy_list[sim_idx] = entropy(statistics.trans_to_categorical(estimation_1), base=2)
  mutual_list[sim_idx] = statistics.mutual_info(statistics.trans_to_categorical(estimation_1), statistics.trans_to_categorical(estimation_2))

print("相関係数: ", np.mean(corr_list))
print("エントロピー: ", np.mean(entropy_list))
print("相互情報量: " ,np.mean(mutual_list))