import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import netCDF4 as nc
import xarray as xr


dataset = []
for i in os.listdir("./"):
    if i.split('.')[0].startswith('2023'):
        print(i)
        ds1 = xr.open_dataset(i)

# 创建x, y网格
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()
sc = ax.scatter(X[::5, ::5], Y[::5, ::5], c=[], cmap='viridis', vmin=-2, vmax=2) # 使用颜色表示z值
plt.colorbar(sc)

def init():
    sc.set_array([])
    return sc,

def update(frame):
    Z = np.sin(X + frame/10) + np.cos(Y + frame/10) # 示例z数据
    sc.set_array(Z[::5, ::5].ravel())
    return sc,

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=False)

# plt.show()