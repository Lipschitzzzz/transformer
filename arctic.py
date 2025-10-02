import xarray as xr

filename = "./dataset/20230924120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
ds = xr.open_dataset(filename)
lons = ds['lon'].values
lats = ds['lat'].values

print(ds.variables)