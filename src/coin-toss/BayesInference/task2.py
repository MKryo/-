# 非定常環境
import model
import numpy as np
import matplotlib.pyplot as plt

def generate_sin(T):
  return (np.sin(timestep/400)+1)/2

def generate_sample(number_of_samples, p):  # サンプルデータの作成
  r = np.random.rand(number_of_samples) #乱数を試行回数分だけ生成
  d = r < p
  return d.astype(np.int8)

# 共通設定
N=10 # 仮説数
H_likelihoods = np.arange(0, 1.1, 1/N)  # 表尤度 C(H | h_i)
T_likelihoods = 1 - H_likelihoods  # 裏尤度 C(T | h_i)
prior_prob = np.array([1/11]*len(H_likelihoods)) # 事前分布： 一様分布

# 表率 sin 設定
TOSS_NUM = 16000 # 試行回数
timestep = np.arange(0,TOSS_NUM)
estimation = np.zeros(TOSS_NUM)  # ベイズ推定量の推移
p_sin = generate_sin(timestep)
l = generate_sample(TOSS_NUM, p_sin)

epsilon = 0
#### ベイズ更新 ####
for i in range(TOSS_NUM):
  prior_prob = model.bayesian_update(prior_prob, H_likelihoods, T_likelihoods, l[i], epsilon)
  estimation[i] = np.sum(prior_prob*H_likelihoods)  # ベイズ推定量の計算 

### plot ###
plt.figure(figsize=(10,5))
plt.title("epsilon 0")
plt.plot(timestep, p_sin, label="correct")
plt.plot(timestep, estimation, label="estimation")
plt.xlabel("time step", fontsize = 15)
plt.ylabel("hypothesis", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 1)
plt.legend()

print("sin: {}".format(l))
MAP = np.argmax(prior_prob)
print("MAP推定値: {}".format(H_likelihoods[MAP]))

epsilon = 1e-5
h = np.zeros((N+1, TOSS_NUM))
H_h = np.zeros((N+1, TOSS_NUM))
max_h_array = np.zeros(TOSS_NUM)

#### ベイズ更新 ####
for i in range(TOSS_NUM):
  prior_prob = model.bayesian_update(prior_prob, H_likelihoods, T_likelihoods, l[i], epsilon)
  estimation[i] = np.sum(prior_prob*H_likelihoods)  # ベイズ推定量の計算

  ### plot 用 ###
  for j in range(N+1):
    h[j][i] = prior_prob[j]
    H_h[j][i] = H_likelihoods[j]
  max_h_array[i] = model.max_hypothesis(prior_prob)
  #############

### plot ###
plt.figure(figsize=(10,5))
plt.title("epsilon 1e-5")
plt.plot(timestep, p_sin, label="correct")
plt.plot(timestep, estimation, label="estimation")
plt.xlabel("time step", fontsize = 15)
plt.ylabel("hypothesis", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 1)
plt.legend()

print("sin: {}".format(l))
MAP = np.argmax(prior_prob)
print("MAP推定値: {}".format(H_likelihoods[MAP]))

### plot ###
plt.figure(figsize=(12,5))
for i in range(N+1):
  plt.plot(timestep, h[i], label="h"+str(i))
plt.xlabel("time step", fontsize = 15)
plt.ylabel("C(h)", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 1)
plt.legend()

### plot ###
plt.figure(figsize=(12,5))
plt.plot(timestep, max_h_array, label="hypothesis with maximum confidence h")
plt.xlabel("time step", fontsize = 15)
plt.ylabel("hypothesis with maximum confidence h", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 10)
plt.legend()

### plot ###
plt.figure(figsize=(12,5))
for i in range(N+1):
  plt.plot(timestep, H_h[i], label="h"+str(i))
plt.xlabel("time step", fontsize = 15)
plt.ylabel("C(H|h)", fontsize = 15)
plt.xlim(0, TOSS_NUM)
plt.ylim(0, 1)
plt.legend()
plt.show()