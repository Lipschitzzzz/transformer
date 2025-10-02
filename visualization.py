import netCDF4 as nc
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

filename = "./dataset_arctic_sst/20250501120000-DMI-L4_GHRSST-STskin-DMI_OI-ARC_IST-v02.0-fv01.0.nc"

f = nc.Dataset(filename)

all_vars = f.variables.keys()
# 如果经度是 0~360，想转为 -180~180

# 注意：这取决于你的数据结构，可能需要调整
# valid_data_mask = ~np.isnan(filtered_data) & ~np.isinf(filtered_data)
# filtered_dataset = dataset[valid_data_mask]
# 设置统一的颜色范围（可选，保持图颜色一致）
# filtered_data -= offset


time = []
# print(all_vars)
# all_vars_info = f.variables.items()
# print(all_vars_info)
n_end = 640
n_start = 0
e_end = 7200
e_start = 0
dataset2 = []
ds1 = xr.open_dataset(filename)
lats = ds1['lat'].values
lons = ds1['lon'].values
lons = (lons + 180) % 360 - 180
for i in os.listdir("./dataset_arctic_sst"):
    if i.split('.')[0].startswith('2025'):
        print(i)
        ds1 = xr.open_dataset("./dataset_arctic_sst/" + i)
        dataset = ds1['analysed_st'][0].values
        # filtered_data = dataset[n_start:n_end, e_start:e_end]
        filtered_data = dataset
        dataset2.append(filtered_data)
        time.append(i.split('-')[0])
vmin = np.nanmin(dataset2)
vmax = np.nanmax(dataset2)
dataset2 = np.array(dataset2)
print(dataset2.shape)
# all_vars_info = list(all_vars_info)

print("Min value:", np.nanmin(dataset))
print("Max value:", np.nanmax(dataset))

fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
def add_features(ax):
    """添加地理特征"""
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.gridlines(draw_labels=True)
    ax.set_extent([lons[e_start], lons[e_end - 1], lats[n_start], lats[n_end - 1]], crs=ccrs.PlateCarree())
lo = lons[e_start:e_end]
la = lats[n_start:n_end]
z = filtered_data
def plot_map(ax, data, title):
    """绘图函数"""
    c = ax.contourf(lo, la, data, levels=20, cmap='viridis', vmin=vmin, vmax=vmax, transform=ccrs.Orthographic(0, 90))
    ax.set_title(title)
    add_features(ax)
    return c

# 绘制单个地图
# c = plot_map(ax, filtered_data, "Regional Ocean SSH GLORYS12 Reanalysis")

cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.02])  # 调整了底部位置以适应标题和其他元素

ax.set_title(f"Sea Surface Temperature - Frame 0")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, color='lightgray')
ax.gridlines(draw_labels=True)
c = ax.pcolormesh(lo, la, dataset2[0], 
                     cmap='viridis', 
                     vmin=vmin, 
                     vmax=vmax, 
                     shading='auto', 
                     transform=ccrs.Orthographic(0, 90))
cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal', label='Value Label')  # 更新了颜色条标签
# 设置地图显示范围
# ax.set_extent([e_start, e_end, n_start, n_end], crs=ccrs.PlateCarree())

# 添加颜色条
# cbar = plt.colorbar(c, ax=ax, orientation='horizontal', label='SST (K)', pad=0.05)

# 更新函数
def update(frame):
    c.set_array(dataset2[frame].ravel())  # pcolormesh 要求展平的数组
    ax.set_title(f"Sea Surface Temperature - {time[frame]}")
    return [c]
from matplotlib.animation import FuncAnimation

# 创建动画
# ani = FuncAnimation(fig, update, frames=len(dataset2), 
#                     init_func=None, 
#                     blit=False, 
#                     interval=200)
# ani.save('SST_Animation.gif', writer='pillow', fps=10, dpi=100)

print("SST_Animation.gif")
# 自动调整布局
# plt.tight_layout(rect=[0, 0.1, 1, 1])  # 留出底部空间给颜色条
# def init():
#     c.set_array([])
#     return c,

# def update(frame):
#     # 清除当前内容
#     ax.clear()
#     add_features(ax)
#     # 重新绘制 contourf
#     ax.contourf(lo, la, dataset2[frame], levels=20, cmap='viridis', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
#     ax.set_title(f"Frame {frame}")
#     return []

# ani = FuncAnimation(fig, update, frames=len(dataset2), init_func=init, blit=False)

# plt.show()
# print(lons[6000:6400])
# print(lats[0])
# print(lats[3600-1])
# print(lons[0])
# print(lons[7200-1])
# plt.savefig('Korean sea.jpg')

# 显示图像
plt.show()
