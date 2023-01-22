import numpy as np

def bayes_update(prior_probs, H_likelihoods, T_likelihoods, d, alpha, m, epsilon):

  likelihoods = (H_likelihoods)**d * (T_likelihoods)**(1-d)
  posterior_probs = likelihoods * prior_probs
  posterior_probs_copy = np.copy(posterior_probs)
  denom = ( ((1-alpha)*np.sum(posterior_probs+epsilon)**(-m)) + (alpha*(prior_probs)**(-m)) ) ** (-1/m)
  posterior_probs = (posterior_probs+epsilon) / denom # (25)式
  return posterior_probs / np.sum(posterior_probs) , np.sum(posterior_probs_copy) # 正規化 (29)式
  
def Inverse_bayes_update(prior_probs, H_likelihoods, T_likelihoods, d, alpha, m, epsilon, C_d):
  # もし最も確信度の高い仮説が複数あったら、ランダムに一つ選ぶ
  # max_hypo_idx = np.argwhere(prior_probs == np.amax(prior_probs))
  # hoge = np.array(max_hypo_idx.flatten().tolist())
  # max_h = np.random.choice(hoge, 1)
  # print(max_h)
  max_h = np.random.choice(np.where(prior_probs == prior_probs.max())[0])
  
  if d==1:

    H_likelihoods[max_h] = prior_probs[max_h] * H_likelihoods[max_h]
    denom = ( ((1-alpha)*(prior_probs[max_h]) **(-m)) + (alpha * C_d **(-m)) ) ** (-1/m)
    H_likelihoods[max_h] = H_likelihoods[max_h] / denom # (28)式
    norm = H_likelihoods[max_h] + T_likelihoods[max_h] # 正規化項
    
    H_likelihoods[max_h] = H_likelihoods[max_h] / norm # C(H|h_max) 正規化 (29)式
    T_likelihoods[max_h] = T_likelihoods[max_h] / norm # C(T|h_max) 正規化 (29)式
    return H_likelihoods, T_likelihoods, max_h
    
  elif d==0:

    T_likelihoods[max_h] = prior_probs[max_h] * T_likelihoods[max_h]
    denom = ( ((1-alpha)*(prior_probs[max_h]) **(-m)) + (alpha * C_d **(-m)) ) ** (-1/m)
    T_likelihoods[max_h] = T_likelihoods[max_h] / denom # (28)式
    norm = H_likelihoods[max_h] + T_likelihoods[max_h] # 正規化項
    
    H_likelihoods[max_h] = H_likelihoods[max_h] / norm # # C(H|h_max) 正規化 (29)式
    T_likelihoods[max_h] = T_likelihoods[max_h] / norm # # C(T|h_max) 正規化 (29)式
    return H_likelihoods, T_likelihoods, max_h