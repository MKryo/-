import numpy as np

grid = np.arange(-2, 2, 0.1)
alpha_m_column = np.array([0,0,0])
for k in range(len(grid)):
    alpha = grid[k]
    for j in range(len(grid)):
        m = grid[j]
        print(alpha)
        print(m)
        mutual = 0.99
        alpha_m_column = np.vstack([alpha_m_column, np.array([alpha, m, mutual])])

np.savetxt('savetxt.csv',alpha_m_column ,delimiter=',')