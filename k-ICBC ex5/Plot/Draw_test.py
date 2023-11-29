import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


fig = plt.figure(
    figsize = (10, 8),
    dpi = 200
)

ax = fig.add_subplot()
ax.grid()

# X全域
ax.fill_between([-np.pi/2, np.pi/2], -np.pi/2, np.pi/2, facecolor='gray', alpha = 0.5)

# Xi
ax.fill_between([-np.pi/9, np.pi/9], -np.pi/9, np.pi/9, facecolor='green', alpha=0.4)

# Xu
ax.fill_between([-np.pi/2, -np.pi/6], -np.pi/2, -np.pi/6, facecolor='red', alpha=0.4)
ax.fill_between([-np.pi/2, -np.pi/6], np.pi/6, np.pi/2, facecolor='red', alpha=0.4)
ax.fill_between([np.pi/6, np.pi/2], -np.pi/2, -np.pi/6, facecolor='red', alpha=0.4)
ax.fill_between([np.pi/6, np.pi/2], np.pi/6, np.pi/2, facecolor='red', alpha=0.4)
plt.show()