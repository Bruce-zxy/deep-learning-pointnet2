import random
import h5py
import numpy as np

with h5py.File('./plant_train.h5', 'r') as f:
    data_full = f['data'][()]
    print(np.max(np.sqrt(np.sum(abs(data_full[20])**2, axis=-1))))

    for key in f.keys():
        print(f[key])
