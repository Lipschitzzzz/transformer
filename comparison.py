import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

# 使用等经纬度投影（适合小区域）
proj = ccrs.PlateCarree()

# 创建图形和两个子图（1行2列）
fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': proj})

# 设置朝鲜半岛的地理范围
# extent = [120, 135, 30, 45]
extent = [0, 180, 0, 90]

# 共享的地理特征
for ax in axes:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='white')

# 加载经纬度
filename = "./dataset/20230925120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
ds = xr.open_dataset(filename)
lons = ds['lon'].values
lats = ds['lat'].values
data_train = ds['analysed_sst'].values - 273.15
# 提取区域索引
n_start, n_end = 2400, 2700
e_start, e_end = 6000, 6300
lo = lons[e_start:e_end]
la = lats[n_start:n_end]
lon_grid, lat_grid = np.meshgrid(lo, la)

# 加载数据
# ds_train = np.load("Training.npz")['data']
ds_output = np.load("output300.npz")['data']

# 提取对应区域的数据（注意：假设 data shape 是 [H, W]）
data_output = ds_output[0] # 预测值

print(data_train[0].shape)
print(data_output.shape)
# 设置统一 colorbar 范围（推荐，便于对比）
vmin = np.nanmin([data_train[0, n_start:n_end, e_start:e_end],data_output])
vmax = np.nanmax([data_train[0, n_start:n_end, e_start:e_end],data_output])

# --- 左图：真实数据 ---
c1 = axes[0].pcolormesh(lons, lats, data_train[0],
                        cmap='viridis', vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), shading='auto')
axes[0].set_title("Sea Surface Temperature - Ground Truth", fontsize=14)
# data_train[0, n_start:n_end, e_start:e_end]
# --- 右图：预测数据 ---
c2 = axes[1].pcolormesh(lon_grid, lat_grid, data_output,
                        cmap='viridis', vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), shading='auto')
axes[1].set_title("Sea Surface Temperature - Predicted", fontsize=14)

# 添加统一的颜色条
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(c1, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Temperature (°C)')

# 调整布局
plt.tight_layout()
plt.show()