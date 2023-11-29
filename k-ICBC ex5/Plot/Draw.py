import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from Loss_Encoding import LossHandler
import superp

# 无法绘制图像
'''
def path_simulation(fx_, control_model):

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

    # ——————————————————————————————————起始点——————————————————————————————————
    k_ = 5000

    # 初始区域取点
    x_data = np.linspace(-np.pi, np.pi, 10)
    y_data = np.linspace(-5, 5, 10)

    s = np.array(np.meshgrid(x_data, y_data))
    b = s.reshape(-1, order='F')
    s = b.reshape(-1, 2)
    x = s[:, 0]
    y = s[:, 1]
    con = x ** 2 + y ** 2 <= 4
    s = s[con]


    x0 = torch.tensor(s, dtype=torch.float64).to(superp.device)

    result = LossHandler.calc_K_iteration(x0, fx_, control_model, k_)
    result = torch.cat(result, 1).cpu().detach().numpy()

    for i in range(result.shape[0]):
        route = result[i, :].reshape((-1, 2))
        ax.scatter(route[:, 0], route[:, 1], alpha=1, s=1)

    plt.show()



'''
