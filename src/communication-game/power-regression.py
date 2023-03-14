import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('L_data.csv')

df = df[df['y0'] > 1e-6]
df['X-value'] = df['x0']
df['Y-value'] = df['y0']

#自然対数をとる
df['X-value_log'] = np.log(df['X-value'])
df['Y-value_log'] = np.log(df['Y-value'])

#変数を指定。[]が2つあるのは、sklearnとPandasの関係性のため。
X = df[['X-value_log']]
Y = df[['Y-value_log']]

#単回帰分析
reg = LinearRegression().fit(X, Y)

#予測結果をdfに追加
df['pred'] = np.exp(reg.intercept_) * df['X-value'] **reg.coef_[0, 0]

print("回帰係数: ",reg.coef_)
print("切片: ",reg.intercept_)
print("決定係数: ",reg.score(X, Y))

a, figure = plt.subplots()
figure.set_xscale('log')
figure.set_yscale('log')
# plt.xlim(1,2000)
plt.xlabel('X-value')
plt.ylabel('Y-value')
plt.scatter(df['X-value'], df['Y-value'])
plt.plot(df['X-value'], df['pred'], color='orange', label="reg_line")
plt.legend()
plt.show()

