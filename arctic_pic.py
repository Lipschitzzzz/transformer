import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.patches import Circle
from matplotlib.path import Path

# 创建北极立体投影
proj = ccrs.NorthPolarStereo(central_longitude=0)  # 可改为 180 或其他

# 创建图形和轴
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': proj})

# 设置范围：限制在北纬 60° 以上（可调）
ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

# 添加地理特征
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN, color='white')

# === 绘图：使用 pcolormesh 或 contourf ===
# 假设你有：
#   lo: 经度网格 (lon), shape: (H, W)
#   la: 纬度网格 (lat), shape: (H, W)
#   filtered_data: 数据矩阵, shape: (H, W)
import xarray as xr
filename = "./dataset_arctic_sst/20250501120000-DMI-L4_GHRSST-STskin-DMI_OI-ARC_IST-v02.0-fv01.0.nc"

ds1 = xr.open_dataset(filename)
lats = ds1['lat'].values
lons = ds1['lon'].values
dataset = ds1['analysed_st'][0].values
print(dataset.shape)
print(lats.shape)
print(lons.shape)

vmin = np.nanmin(dataset)
vmax = np.nanmax(dataset)
lons = (lons + 180) % 360 - 180
n_end = 640
n_start = 0
e_end = 7200
e_start = 0
lo = lons[e_start:e_end]
la = lats[n_start:n_end]
print(dataset)
print(la)
print(lo)
# lons = (lons + 180) % 360 - 180
# 设置地图显示范围为实际经纬度范围
ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
# 关键：使用 transform=ccrs.PlateCarree() 将 lon/lat 转换到极射投影
# c = ax.contourf(lo, la, dataset[n_start:n_end, e_start:e_end],
#                 levels=20,
#                 cmap='viridis',
#                 vmin=vmin, vmax=vmax,
#                 transform=ccrs.PlateCarree())
c = ax.pcolormesh(lo, la, dataset[n_start:n_end, e_start:e_end],
                  cmap='viridis', vmin=vmin, vmax=vmax,
                  transform=ccrs.PlateCarree(), shading='auto')
# === 标题 ===
ax.set_title(f"Sea Surface Temperature - Frame 0", fontsize=14)

# === 添加颜色条 ===
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Temperature (°C)')

# 显示
plt.show()