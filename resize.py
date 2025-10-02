import numpy as np
with np.load("Training.npz") as data:
    arr = data['data']
    n_start = 2400
    n_end = 2700
    e_start = 6000
    e_end = 6300
    arr = arr[:, n_start:n_end, e_start:e_end] - 273.15
    print(arr.shape)
    np.savez_compressed("Training300.npz", data = arr)

        