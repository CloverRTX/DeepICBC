import numpy as np
import matplotlib.pyplot as plt

import Sampling.getTrainingData


if __name__ == "__main__":
    pass

fig = plt.figure(
    figsize=(10, 8),
    dpi=200
)

ax = fig.add_subplot()
ax.grid()

# ——————————————————————————————————Plot——————————————————————————————————

# X全域
ax.fill_between([-4., 4.], -4., 4., facecolor='gray', alpha=0.5)

# Xi
ax.fill_between([-3, -1], -3., -1., facecolor='green', alpha=0.4)

# Xu
sita = np.arange(0, 2*np.pi, 0.05)
ax.fill_between(np.cos(sita) + 3, np.sin(sita) + 2,facecolor='red', alpha=0.4)
ax.axis('equal')

# 边界域
ax.fill_between([-5, 5], -5, 5, facecolor='yellow', alpha=0.1)



data_Xi, data_Xu, data_X, data_X_bounded_area = Sampling.getTrainingData.Sample_Handler.getTrainingData()

#ax.scatter(data_Xi.iloc[:, 0], data_Xi.iloc[:, 1], alpha=1, s=1)
#ax.scatter(data_Xu.iloc[:, 0], data_Xu.iloc[:, 1], alpha=1, s=1)
#ax.scatter(data_X.iloc[:, 0], data_X.iloc[:, 1], alpha=1, s=1)
#ax.scatter(data_X_bounded_area.iloc[:, 0], data_X_bounded_area.iloc[:, 1], alpha=1, s=1)


plt.show()

