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

# ——————————————————————————————————绘制动力系统的全局图像——————————————————————————————————

# X全域
ax.fill_between([-np.pi, np.pi], -5, 5, facecolor='gray', alpha=0.5)

seta = np.arange(0, 2 * np.pi, 0.01)
# print(seta.shape)
temp_x = np.zeros((seta.shape[0],))
temp_y = np.zeros((seta.shape[0],))

# Xi
ax.fill_between(2 * np.cos(seta), 0, 2 * np.sin(seta), facecolor='green', alpha=0.4)

# Xu
x1 = np.arange(-3, -2.5, 0.01)
x2 = np.arange(-2.5, 2.5, 0.01)
x3 = np.arange(2.5, 3, 0.01)

ax.fill_between(x1, -np.sqrt(9 - x1 ** 2), np.sqrt(9 - x1 ** 2), facecolor='red', alpha=0.4)
ax.fill_between(x2, np.sqrt(2.5 ** 2 - x2 ** 2), np.sqrt(9 - x2 ** 2), facecolor='red', alpha=0.4)
ax.fill_between(x2, -np.sqrt(9 - x2 ** 2), -np.sqrt(2.5 ** 2 - x2 ** 2), facecolor='red', alpha=0.4)
ax.fill_between(x3, -np.sqrt(9 - x3 ** 2), np.sqrt(9 - x3 ** 2), facecolor='red', alpha=0.4)

ax.axis('equal')



# ——————————————————————————————————验证采样点——————————————————————————————————

data_Xi, data_Xu, data_X = Sampling.getTrainingData.Sample_Handler.getTrainingData()

#ax.scatter(data_Xi.iloc[:, 0], data_Xi.iloc[:, 1], alpha=1, s=10)
#ax.scatter(data_Xu.iloc[:, 0], data_Xu.iloc[:, 1], alpha=1, s=10)
#ax.scatter(data_X.iloc[:, 0], data_X.iloc[:, 1], alpha=1, s=10)

#point = [-2.305981055555579, -0.9656349593964066]
#ax.scatter(point[0], point[1], alpha=1, s=10)

plt.show()

