import numpy as np
import matplotlib.pyplot as plt
import model

def generate_sample(number_of_samples, p):  # サンプルデータの作成
  r = np.random.rand(number_of_samples) #乱数を試行回数分だけ生成
  d = r < p
  return d.astype(np.int8)

def generate_sin(T):
  return (np.sin(timestep/400)+1)/2

# 共通設定
N=10
H_likelihoods = np.arange(0, 1.1, 1/N) # 表の尤度 C(d=1|h_max) (31)式
T_likelihoods = 1 - H_likelihoods # 裏の尤度 C(d=0|h_max) (31)式
prior_probs = np.array([1/11]*len(H_likelihoods)) # 事前分布： 一様分布 (32)式

TOSS_NUM = 16000 # 試行回数
N=5
H_likelihoods = np.arange(0, 0.6, 0.5/N) # 表の尤度 C(d=1|h_max) (31)式
T_likelihoods = 1 - H_likelihoods # 裏の尤度 C(d=0|h_max) (31)式
prior_probs = np.array([1/(N+1)]*len(H_likelihoods)) # 事前分布： 一様分布 (32)式
timestep = np.arange(0,TOSS_NUM) # 横軸
estimation = np.zeros(TOSS_NUM)  # ベイズ推定量の推移
p_sin = generate_sin(timestep)
l = generate_sample(TOSS_NUM, p_sin)
epsilon = 0

# ハイパーパラメータ
alpha = 0.21
m = -1.0

#### ベイズ更新 ####
for i in range(TOSS_NUM):
  # 退避が必要
  prior_probs_copy = np.copy(prior_probs)
  prior_probs , C_d= model.bayes_update(prior_probs, H_likelihoods, T_likelihoods,  l[i], alpha, m, epsilon)
  H_likelihoods, T_likelihoods, max_h = model.Inverse_bayes_update(prior_probs_copy, H_likelihoods, T_likelihoods, l[i], alpha, m, epsilon, C_d)
  estimation[i] = np.sum(prior_probs*H_likelihoods)  # ベイズ推定量の計算

### plot ###
plt.figure(figsize=(12,5))
plt.plot(timestep, p_sin, label="correct")
plt.plot(timestep, estimation, label="estimation")
plt.xlabel("time step", fontsize = 15)
plt.ylabel("hypothesis", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 1)
plt.legend()
plt.show()