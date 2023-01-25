# ライブラリ
import numpy as np
import model
import statistics

SIM_NUM=100
alpha_list = np.zeros(SIM_NUM)
m_list = np.zeros(SIM_NUM)

for sim_idx in range(SIM_NUM):
    mutual_info_list=[]
    grid_alpha = np.arange(0, 1.0, 0.1)
    grid_m = np.arange(-2.0, 2.0, 0.1)
    alpha_m_mutual_column = np.array([0,0,0])
    m=0
    for k in range(len(grid_alpha)):
        alpha = grid_alpha[k]
        for j in range(len(grid_m)):
            m = grid_m[j]
            # print(alpha)
            # print(m)
            # 共通設定
            K=0 # 仮説数K+1
            H_likelihoods_1 = np.random.rand(K+1) # 尤度の初期値ランダム
            T_likelihoods_1 = 1 - H_likelihoods_1 # 裏の尤度 C(d=0|h_max) (31)式
            H_likelihoods_2 = np.random.rand(K+1) # 尤度の初期値ランダム
            T_likelihoods_2 = 1 - H_likelihoods_2 # 裏の尤度 C(d=0|h_max) (31)式
            prior_probs_1 = np.array([1/(K+1)]*len(H_likelihoods_1)) # 事前分布： 一様分布 (32)式
            prior_probs_2 = np.array([1/(K+1)]*len(H_likelihoods_2)) # 事前分布： 一様分布 (32)式
            d1= 1 if np.random.rand() < 0.9 else 0
            d2= 1 if np.random.rand() < 0.9 else 0
            agent1 = model.Agent(K, alpha, m)
            agent2 = model.Agent(K, alpha, m)
            ## シミュレーション ##
            TOSS_NUM=20000 # 試行回数
            timestep = np.arange(0,TOSS_NUM) # 横軸
            estimation_1 = np.zeros(TOSS_NUM)  # ベイズ推定量の推移
            estimation_2 = np.zeros(TOSS_NUM)  # ベイズ推定量の推移
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
            mutual = statistics.mutual_info(statistics.trans_to_categorical(estimation_1), statistics.trans_to_categorical(estimation_2))
            mutual_info_list.append(mutual)
            alpha_m_mutual_column = np.vstack([alpha_m_mutual_column, np.array([alpha, m, mutual])])
            m+=1
    # np.savetxt('info.csv',alpha_m_mutual_column ,delimiter=',')
    mutual_info_list = np.array(mutual_info_list)
    # print(mutual_info_list)
    # print(alpha_m_mutual_column[np.argmax(mutual_info_list)][0])
    # print(alpha_m_mutual_column[np.argmax(mutual_info_list)][1])

    alpha_list[sim_idx] = alpha_m_mutual_column[np.argmax(mutual_info_list)][0]
    m_list[sim_idx] = alpha_m_mutual_column[np.argmax(mutual_info_list)][1]

    mutual_info_list = list(mutual_info_list)

print(np.mean(alpha_list, axis=0))
print(np.mean(m_list, axis=0))
