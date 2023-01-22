import numpy as np

def max_hypothesis(prior_prob):
  # もし最も確信度の高い仮説が複数あったら、ランダムに一つ選ぶ
  max_hypo_idx = np.argwhere(prior_prob == np.amax(prior_prob))
  hoge = np.array(max_hypo_idx.flatten().tolist())
  max_h = np.random.choice(hoge, 1)
  return max_h[0]

def bayesian_update(prior_prob, H_likelihoods, T_likelihoods, d, epsilon):
  posterior_prob = (H_likelihoods**d * (T_likelihoods)**(1-d)) * prior_prob # 尤度*事前分布
  norm = np.sum(posterior_prob+epsilon)  # 規格化項
  return (posterior_prob+epsilon) / norm
