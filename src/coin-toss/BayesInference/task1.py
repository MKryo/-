# 定常環境
import model
import numpy as np
import matplotlib.pyplot as plt

def generate_sample(number_of_samples, p):  # サンプルデータの作成
  r = np.random.rand(number_of_samples) #乱数を試行回数分だけ生成
  d = r < p
  return d.astype(np.int8)

# 共通設定
N=10 # 仮説数
H_likelihoods = np.arange(0, 1.1, 1/N)  # 表尤度 C(H | h_i)
T_likelihoods = 1 - H_likelihoods  # 裏尤度 C(T | h_i)
prior_prob = np.array([1/11]*len(H_likelihoods)) # 事前分布： 一様分布

# task1
TOSS_NUM=4000 # 試行回数
p=0.8 # 表確率
l = generate_sample(TOSS_NUM, p) # データ生成
print("真の表率",p)

timestep = np.arange(0,TOSS_NUM)
estimation = np.zeros(TOSS_NUM)  # ベイズ推定量の推移
epsilon = 0

#### ベイズ更新 ####
for i in range(TOSS_NUM):
  prior_prob = model.bayesian_update(prior_prob, H_likelihoods, T_likelihoods, l[i], epsilon)
  estimation[i] = np.sum(prior_prob*H_likelihoods)  # ベイズ推定量の計算

### plot ###
plt.figure(figsize=(10,5))
correct = np.full(TOSS_NUM ,p)
plt.title("epsilon 0")
plt.plot(timestep, correct, label="correct")
plt.plot(timestep, estimation, label="estimation")
plt.xlabel("time step", fontsize = 15)
plt.ylabel("H_likelihoods", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 1)
plt.legend()

MAP = np.argmax(prior_prob)
print("MAP推定値: {}".format(H_likelihoods[MAP]))

TOSS_NUM=4000 # 試行回数
p=0.8 # 表確率
l = model.generate_sample(TOSS_NUM, p)
print("真の表率",p)

timestep = np.arange(0,TOSS_NUM)
estimation = np.zeros(TOSS_NUM)  # ベイズ推定量の推移
epsilon = 1e-5

#### ベイズ更新 ####
for i in range(TOSS_NUM):
  prior_prob = model.bayesian_update(prior_prob, H_likelihoods, T_likelihoods, l[i], epsilon)
  estimation[i] = np.sum(prior_prob*H_likelihoods)  # ベイズ推定量の計算

### plot ###
plt.figure(figsize=(10,5))
correct = np.full(TOSS_NUM ,p)
plt.title("epsilon 1e-5")
plt.plot(timestep, correct, label="correct")
plt.plot(timestep, estimation, label="estimation")
plt.xlabel("time step", fontsize = 15)
plt.ylabel("hypothesis", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 1)
plt.legend()
plt.show()

# print("coin list: {}".format(l))
MAP = np.argmax(prior_prob)
print("MAP推定値: {}".format(H_likelihoods[MAP]))