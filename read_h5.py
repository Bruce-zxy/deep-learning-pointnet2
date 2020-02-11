import h5py

with h5py.File('./plant_train.h5', 'r') as f:
    for key in f.keys():
        print(f[key])