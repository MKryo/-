import numpy as np

class Agent:
  def __init__(self, K, alpha, m):
    self.K=K
    self.epsilon=0
    self.alpha=alpha
    self.m=m

  def generator(self, p):  # サンプルデータの作成
    r = np.random.rand()
    return 1 if r < p else 0

  def bayes_update(self, prior_probs, H_likelihoods, T_likelihoods, d):
    likelihoods = ((H_likelihoods)**d) * ((T_likelihoods)**(1-d))
    posterior_probs = likelihoods * prior_probs
    return np.sum(posterior_probs) # 正規化 (29)式

  def Inverse_bayes_update(self, prior_probs, C_d_H, C_d_T, d, C_d):
    if d==1:

      denom = ( (1-self.alpha) + (self.alpha * C_d **(-self.m)) ) ** (-(1/self.m))
      C_d_H = C_d / denom # (28)式
      norm = C_d_H + C_d_T# 正規化項

      C_d_H = C_d_H / norm # C(H|h_max) 正規化 (29)式
      C_d_T = C_d_T / norm # C(T|h_max) 正規化 (29)式
      return C_d_H, C_d_T

    elif d==0:

      denom = ( (1-self.alpha) + (self.alpha * C_d **(-self.m)) ) ** (-(1/self.m))
      C_d_T = C_d_T / denom # (28)式
      norm = C_d_H + C_d_T # 正規化項

      C_d_H = C_d_H / norm # C(H|h_max) 正規化 (29)式
      C_d_T = C_d_T / norm # C(T|h_max) 正規化 (29)式
      return C_d_H, C_d_T