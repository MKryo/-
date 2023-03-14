import numpy as np


grid_alpha = np.arange(0.0, 1.0, 0.05)
grid_m = np.arange(-2.0, 2.0, 0.1)
out=[]

for k in range(len(grid_alpha)):
        alpha = grid_alpha[k]
        for j in range(len(grid_m)):
            m = grid_m[j]
            out.append(np.array([alpha, m]))
out = np.array(out)
np.savetxt('out.csv', out)