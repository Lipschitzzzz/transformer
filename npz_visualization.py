import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

# 使用等经纬度投影（适合小区域）
proj = ccrs.PlateCarree()

# 创建图形和轴
fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': proj})

# 设置朝鲜半岛的地理范围
# 经度：124°E - 132°E，纬度：34°N - 42°N
ax.set_extent([120, 135, 30, 45], crs=ccrs.PlateCarree())

# 添加地理特征
ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1)
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.OCEAN, color='white')

# 加载原始 nc 文件获取经纬度
filename = "./dataset/20230924120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
ds = xr.open_dataset(filename)
lons = ds['lon'].values
lats = ds['lat'].values

# 提取朝鲜半岛区域的经纬度子集
n_start = 2400
n_end = 2700
e_start = 6000
e_end = 6300
lo = lons[e_start:e_end]
la = lats[n_start:n_end]
print(lo)
print(la)
# 构造网格
lon_grid, lat_grid = np.meshgrid(lo, la)

# 加载预测数据
ds2 = np.load("Training.npz")['data']
data = ds2[0]  # 取第一帧

print(data[n_start:n_end, e_start:e_end].shape)
# 处理 NaN 值
vmin = np.nanmin(data[n_start:n_end, e_start:e_end])
vmax = np.nanmax(data[n_start:n_end, e_start:e_end])

# 绘制 pcolormesh
c = ax.pcolormesh(lon_grid, lat_grid, data[n_start:n_end, e_start:e_end],
                  cmap='viridis', vmin=vmin, vmax=vmax,
                  transform=ccrs.PlateCarree(), shading='auto')

ax.set_title("Sea Surface Temperature - (Korean Peninsula)", fontsize=14)

# 调整颜色条位置
cbar_ax = fig.add_axes([0.25, 0.07, 0.5, 0.02])
cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Temperature (°C)')

plt.show()