# ライブラリ
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../data/sim1_100000timestep_m=-1.1.csv')
a, figure = plt.subplots()
figure.plot(df['x0'],df['y0'],label="alpha=0.0")
figure.plot(df['x0.1'],df['y0.1'],label="alpha=0.1")
figure.plot(df['x0.3'],df['y0.3'],label="alpha=0.3")
figure.plot(df['x0.5'],df['y0.5'],label="alpha=0.5")
figure.set_xscale('log')
figure.set_yscale('log')
plt.xlabel("Length", fontsize = 10)
plt.ylabel("Cumulative relative frequency", fontsize = 10)
plt.legend()

a, figure = plt.subplots()
figure.plot(df['x0'],df['y0'],label="alpha=0.0")
figure.plot(df['x0.1'],df['y0.1'],label="alpha=0.1")
figure.plot(df['x0.3'],df['y0.3'],label="alpha=0.3")
figure.plot(df['x0.5'],df['y0.5'],label="alpha=0.5")
figure.set_xscale('log')
# figure.set_yscale('log')
plt.xlabel("Length", fontsize = 10)
plt.ylabel("Cumulative relative frequency", fontsize = 10)
plt.legend()
plt.show()