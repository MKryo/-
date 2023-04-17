# ライブラリ
import numpy as np
import matplotlib.pyplot as plt
import model
import statistics

# 共通設定
K=0 # 仮説数K+1
alpha=0
m=0.0001
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
  # estimation_1[i] = H_likelihoods_1[max_h_1] # P(D=表 | h_max)  最尤推定
  # estimation_1[i] = np.random.choice(H_likelihoods_1, p=prior_probs_1) # 事後分布(確信度）に基づく乱数
  d1 = agent1.generator(estimation_1[i]) # generator(表確率の確信度)

  prior_probs_copy_2 = np.copy(prior_probs_2)
  C_d_2 = agent2.bayes_update(prior_probs_2, H_likelihoods_2, T_likelihoods_2, d1)
  H_likelihoods_2, T_likelihoods_2 = agent2.Inverse_bayes_update(prior_probs_copy_2, H_likelihoods_2, T_likelihoods_2, d1, C_d_2)
  estimation_2[i] = np.sum(prior_probs_2*H_likelihoods_2)  # (30)式
  # estimation_2[i] = H_likelihoods_2[max_h_2]
  # estimation_2[i] = np.random.choice(H_likelihoods_2, p=prior_probs_2)
  # print(np.sum(prior_probs_2))
  d2 = agent2.generator(estimation_2[i]) # generator(表確率の確信度)

  ### plot 用 ###
  for j in range(K+1):
    h[j][i] = prior_probs_1[j]
    H_h[j][i] = H_likelihoods_1[j]
  #############

### plot ###
plt.figure(figsize=(20,6))
plt.plot(timestep, estimation_1, label="Agent1")
plt.plot(timestep, estimation_2, label="Agent2")
plt.xlabel("time step", fontsize = 20)
plt.ylabel("estimation of generative model", fontsize = 20)
plt.xlim(0, TOSS_NUM)
plt.tick_params(labelsize=20)
plt.legend(fontsize=20)
plt.savefig("optimal_param_estimation.png",dpi=500)
plt.show()

# 尤度（仮説）の変化
### plot ###
plt.figure(figsize=(15,5))
for i in range(K+1):
  plt.plot(timestep, H_h[i], label="h"+str(i))
plt.xlabel("time step", fontsize = 25)
plt.ylabel("C(H|h)", fontsize = 25)
plt.xlim(0, TOSS_NUM)
plt.tick_params(labelsize=25)
plt.legend(fontsize=40)
plt.legend()
plt.show()

print("相関係数: ", np.corrcoef(estimation_1, estimation_2))
print("相互情報量: " ,statistics.mutual_info(statistics.trans_to_categorical(estimation_1), statistics.trans_to_categorical(estimation_2)))