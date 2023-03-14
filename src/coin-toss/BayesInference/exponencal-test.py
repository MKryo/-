import numpy as np
import matplotlib.pyplot as plt

def generator(p):  # サンプルデータの作成
    r = np.random.rand()
    return 1 if r < p else 0

L_list = np.array([0]*100000)
TOSS_NUM = 100000
coin_list = np.array(["null"]*(TOSS_NUM+1))
coin_list[0] = 0 #0番目にはあらかじめ値を入れちゃう

for i in range(TOSS_NUM):
  L=1 # 0,1が続く長さ
  d = generator(0.9) # generator(表確率の確信度)
  coin_list[i+1] = d
  if coin_list[i] != coin_list[i+1]:
    L_list[L]+=1
    L=1
  else:
    L+=1

x = np.arange(1,TOSS_NUM) # 横軸
y = L_list[1:] / L_list[1] #縦軸
a, figure = plt.subplots()
figure.plot(x, y)
figure.set_xscale('log')
figure.set_yscale('log')
plt.xlabel("Length", fontsize = 10)
plt.ylabel("Cumulative relative frequency", fontsize = 10)
plt.show()