import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from Loss_Encoding import LossHandler
import superp

# 绘制状态点的移动路线
def path_simulation(fx_, control_model):

    fig = plt.figure(
        figsize=(10, 8),
        dpi=200
    )

    ax = fig.add_subplot()
    ax.grid()

    # ——————————————————————————————————绘制动力系统的全局图像——————————————————————————————————

    # X全域
    ax.fill_between([-4., 4.], -4., 4., facecolor='gray', alpha=0.5)

    # Xi
    ax.fill_between([-3, -1], -3., -1., facecolor='green', alpha=0.4)

    # Xu
    sita = np.arange(0, 2 * np.pi, 0.05)
    ax.fill_between(np.cos(sita) + 3, np.sin(sita) + 2, facecolor='red', alpha=0.4)
    ax.axis('equal')

    # 边界域
    ax.fill_between([-5, 5], -5, 5, facecolor='yellow', alpha=0.1)

    # ——————————————————————————————————起始点——————————————————————————————————
    k_ = 10000

    # 初始区域取点
    x_data = np.linspace(-3, -1, 5)
    y_data = np.linspace(-3, -1, 5)

    s = np.array(np.meshgrid(x_data, y_data))
    b = s.reshape(-1, order='F')
    s = b.reshape(-1, 2)


    x0 = torch.tensor(s, dtype=torch.float64).to(superp.device)

    result = LossHandler.calc_K_iteration(x0, fx_, control_model, k_)
    result = torch.cat(result, 1).cpu().detach().numpy()

    for i in range(result.shape[0]):
        route = result[i, :].reshape((-1, 2))
        ax.scatter(route[:, 0], route[:, 1], alpha=1, s=1)

    plt.show()

