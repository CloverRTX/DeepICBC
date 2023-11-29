import numpy as np
import matplotlib.pyplot as plt

import Sampling.getTrainingData


if __name__ == "__main__":
    pass

def draw_Ju(di_x1, di_x2, di_y1, di_y2, di_z1, di_z2, color = 'green',alpha = 0.1):
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



fig = plt.figure(
    figsize=(10, 8),
    dpi=200
)


ax = fig.add_subplot(projection='3d')
ax.grid()

# ——————————————————————————————————绘制动力系统的全局图像——————————————————————————————————


#X_D
draw_Ju(-2.2, 2.2, -2.2, 2.2, -2.2, 2.2, color = 'grey')

#X_0
draw_Ju(-0.2, 0.2, -0.2, 0.2, -0.2, 0.2, color = 'green', alpha = 0.05)

#X_u
'''
draw_Ju(-2.2, 2.2, -2.2, -2, -2.2, 2.2, color='red')
draw_Ju(-2.2, 2.2, 2, 2.2, -2.2, 2.2, color='red')
draw_Ju(2, 2.2, -2, 2, -2.2, 2.2, color='red')
draw_Ju(-2.2, -2, -2, 2, -2.2, 2.2, color='red')
draw_Ju(-2, 2, -2, 2, -2.2, -2, color='red')
draw_Ju(-2, 2, -2, 2, 2, 2.2, color='red')
'''



# ——————————————————————————————————验证采样点——————————————————————————————————

data_Xi, data_Xu, data_X = Sampling.getTrainingData.Sample_Handler.getTrainingData()

#ax.scatter(data_Xi.iloc[:, 0], data_Xi.iloc[:, 1], data_Xi.iloc[:, 2], alpha=1, s=1)
#ax.scatter(data_Xu.iloc[:, 0], data_Xu.iloc[:, 1], data_Xu.iloc[:, 2], alpha=1, s=1)
#ax.scatter(data_X.iloc[:, 0], data_X.iloc[:, 1], data_X.iloc[:, 2], alpha=1, s=1)

#point = [-2.305981055555579, -0.9656349593964066]
#ax.scatter(point[0], point[1], alpha=1, s=10)

plt.show()

