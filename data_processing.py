import numpy as np
import os
import xarray as xr

def nc2npz(path_in, path_out):
    file_list = os.listdir(path_in)
    # print(file_list)
    file_name = ""
    for i in file_list:
        file_name = path_in + "/" + i
        dataset = xr.open_dataset(file_name)
        lats = dataset['lat'].values
        lons = dataset['lon'].values
        dataset = dataset['analysed_sst'][0].values - 273.15
        # print(lats.shape)
        # print(lons.shape)
        # print(dataset.shape)
        data_array = np.array(dataset)
        print(data_array.shape)
        # 读取后清理数据
        arr = np.nan_to_num(data_array, nan=0.0, posinf=1e5, neginf=-1e5)  # 替换 NaN 和 inf
        np.savez_compressed(path_out + "/" + i[0:-3] + ".npz", data = arr)
    
if __name__ == '__main__':
    nc2npz("./dataset/2", "Training Dataset Korean")
    # 加载 npz 文件
    # loaded_data = np.load('./Training Dataset/20250505120000-DMI-L4_GHRSST-STskin-DMI_OI-ARC_IST-v02.0-fv01.0.npz')

    # 获取保存的数据
    # recovered_data_array = loaded_data['data']

    # print("Recovered shape:", recovered_data_array.shape)  # 确认与原始 data_array 相同
    # npz_dir = './Training Dataset'
    # npz_files = os.listdir(npz_dir)
    # if len(npz_files) == 0:
    #     raise ValueError(f"No .npz files found in {npz_dir}")

    # print(f"Loading {len(npz_files)} .npz files...")
    # print(npz_files)

    # data_list = []
    # for npz_file in npz_files:
    #     print(npz_dir + '/' + npz_file)
    #     with np.load(npz_dir + '/' + npz_file) as data:
    #         arr = data['data']
    #         data_list.append(arr)
    # npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    npz_dir = "./Training Dataset Korean"
    npz_files = os.listdir(npz_dir)
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {npz_dir}")

    print(f"Loading {len(npz_files)} .npz files...")
    # print(npz_files)

    data_list = []
    for npz_file in npz_files:
        # print(npz_dir + '/' + npz_file)
        with np.load(npz_dir + '/' + npz_file) as data:
            arr = data['data']
            data_list.append(arr)
    np.savez_compressed("Training.npz", data = data_list)

    # print(dataset.shape)
    # print(lats)
    # print(lons)