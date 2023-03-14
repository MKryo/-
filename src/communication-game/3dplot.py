import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

df = pd.read_csv('../../data/mutual_info_3dplot.csv')
df_causal = pd.read_csv('../../data/fitting_to_confidence_data.csv')

# print(df[:][0])
# print(df[:][1])
# print(df[:][2])

# 3Dプロットを作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.array(df["alpha"])
Y = np.array(df["m"])
Z = np.array(df["mutual_infomation"])

x_new, y_new = np.meshgrid(np.unique(X), np.unique(Y))
z_new = griddata((X, Y), Z, (x_new, y_new))

X = x_new
Y = y_new
Z = z_new
print(X.shape)
print(Y.shape)
print(Z.shape)
my_cmap = plt.get_cmap('binary')
surf = ax.plot_surface(X, Y, Z, cmap = my_cmap, lw=0, rstride=1, cstride=1, alpha=0.7)
fig.colorbar(surf, ax = ax,shrink = 0.5, aspect = 5)
ax.set_xlabel('alpha')
ax.set_ylabel('m')
ax.set_zlabel('mutual_infomation')
plt.savefig("mutual_info_grid.png",dpi=500)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.array(df_causal["alpha"])
Y = np.array(df_causal["m"])
Z = np.array(df_causal["fitting"])
x_new, y_new = np.meshgrid(np.unique(X), np.unique(Y))
z_new = griddata((X, Y), Z, (x_new, y_new))
X = x_new
Y = y_new
Z = z_new
print(X.shape)
print(Y.shape)
print(Z.shape)
my_cmap = plt.get_cmap('binary')
surf = ax.plot_surface(X, Y, Z, cmap = my_cmap, lw=0, rstride=1, cstride=1, alpha=0.7)
ax.set_xlabel('alpha')
ax.set_ylabel('m')
ax.set_zlabel('fitting')
fig.colorbar(surf, ax = ax,shrink = 0.5, aspect = 5)
plt.show()


# # Figureと3DAxeS
# fig = plt.figure(figsize = (8, 8))
# ax = fig.add_subplot(111, projection="3d")
# # 軸ラベルを設定
# ax.set_xlabel("x", size = 16)
# ax.set_ylabel("y", size = 16)
# ax.set_zlabel("z", size = 16)
# # 円周率の定義
# pi = np.pi
# # (x,y)データを作成
# x = np.linspace(-3*pi, 3*pi, 256)
# y = np.linspace(-3*pi, 3*pi, 256)
# # 格子点を作成
# X, Y = np.meshgrid(x, y)
# # 高度の計算式
# Z = np.cos(X/pi) * np.sin(Y/pi)

# # print(X)
# # print(Y)
# print(Z)

# # 曲面を描画
# ax.plot_surface(X, Y, Z, cmap = "summer")
# # 底面に等高線を描画
# ax.contour(X, Y, Z, colors = "black", offset = -1)
# plt.show()