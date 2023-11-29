import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
from Loss_Encoding import LossHandler
import superp


def draw_Ju(ax, di_x1, di_x2, di_y1, di_y2, di_z1, di_z2, color='green', alpha=0.1):
    # Make data
    di_x1_data = np.linspace(di_x1, di_x2, 100)
    di_x2_data = np.linspace(di_y1, di_y2, 100)

    X1, X2 = np.array(np.meshgrid(di_x1_data, di_x2_data))

    points = np.zeros((X1.shape[0], X1.shape[1]))

    hei_button = points + di_z1
    hei_top = points + di_z2

    # Plot the surface
    ax.plot_surface(X1, X2, hei_button, linewidth=0, antialiased=False, alpha=alpha, color=color)
    ax.plot_surface(X1, X2, hei_top, linewidth=0, antialiased=False, alpha=alpha, color=color)

    di_x1_data = np.linspace(di_y1, di_y2, 100)
    di_x2_data = np.linspace(di_z1, di_z2, 100)

    X1, X2 = np.array(np.meshgrid(di_x1_data, di_x2_data))

    points = np.zeros((X1.shape[0], X1.shape[1]))

    hei_button = points + di_x1
    hei_top = points + di_x2
    ax.plot_surface(hei_button, X1, X2, linewidth=0, antialiased=False, alpha=alpha, color=color)
    ax.plot_surface(hei_top, X1, X2, linewidth=0, antialiased=False, alpha=alpha, color=color)

    di_x1_data = np.linspace(di_x1, di_x2, 100)
    di_x2_data = np.linspace(di_z1, di_z2, 100)

    X1, X2 = np.array(np.meshgrid(di_x1_data, di_x2_data))

    points = np.zeros((X1.shape[0], X1.shape[1]))

    hei_button = points + di_y1
    hei_top = points + di_y2
    ax.plot_surface(X1, hei_button, X2, linewidth=0, antialiased=False, alpha=alpha, color=color)
    ax.plot_surface(X1, hei_top, X2, linewidth=0, antialiased=False, alpha=alpha, color=color)
# 绘制状态点的移动路线

def path_simulation(fx_, control_model):


    fig = plt.figure(
        figsize=(10, 8),
        dpi=200
    )

    ax = fig.add_subplot(projection='3d')
    ax.grid()

    # ——————————————————————————————————绘制动力系统的全局图像——————————————————————————————————

    # X_D
    draw_Ju(ax, -2.2, 2.2, -2.2, 2.2, -2.2, 2.2, color='grey')

    # Xi
    draw_Ju(ax, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, color='green')

    # Xu
    '''
    
    draw_Ju(2, 2.2, -2.2, 2.2, -2.2, 2.2, color='red', alpha=0.05)
    draw_Ju(-2.2, -2, -2.2, 2.2, -2.2, 2.2, color='red', alpha=0.05)
    draw_Ju(-2, 2, -2.2, -2, -2.2, 2.2, color='red', alpha=0.05)
    draw_Ju(-2, 2, 2, 2.2, -2.2, 2.2, color='red', alpha=0.05)
    draw_Ju(-2, 2, -2, 2, 2, 2.2, color='red', alpha=0.05)
    draw_Ju(-2, 2, -2, 2, -2.2, -2, color='red', alpha=0.05)
    '''

    #0.2 ,-0.2, 0.2
    # 设置验证步长 k_ = 10
    k_ = 5000

    # 初始区域取点


    x_data = np.linspace(-0.2, 0.2, 4)
    y_data = np.linspace(-0.2, 0.2, 4)
    z_data = np.linspace(-0.2, 0.2, 4)

    s = np.array(np.meshgrid(x_data, y_data, z_data))

    b = s.reshape(-1, order='F')

    s = b.reshape(-1, 3)


    x0 = torch.tensor(s, dtype=torch.float64).to(superp.device)

    result = LossHandler.calc_K_iteration(x0, fx_, control_model, k_)
    result = torch.cat(result, 1).cpu().detach().numpy()

    for i in range(result.shape[0]):
        route = result[i, :].reshape((-1, 3))
        # print(route)
        ax.scatter(route[:, 0], route[:, 1], route[:, 2], alpha=1, s=1)

    plt.show()
