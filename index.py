#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# def getDataFiles(list_filename):
#     return [line.rstrip() for line in open(list_filename)]


# def loadDataFile(path):
#     data = np.loadtxt(path)
#     num = data.shape[0]
#     point_xyz = data[:, 0:3]
#     point_normal = data[:, 3:6]
#     point_rgb = data[:, 6:9]
#     # label just a example, should be repalced the real.
#     # modlenet40 is 0-39, so the label can be 40 and 41
#     label = np.ones((num, 1), dtype=int)+39
#     return point_xyz, label


def change_scale(data):
    #centre
    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min+(xyz_max-xyz_min)/2
    data[:, 0:3] = data[:, 0:3]-xyz_move
    #scale
    scale = np.max(data[:, 0:3])
    data[:, 0:3] = data[:, 0:3]/scale
    return data


# def sample_data(data, num_sample):
#     """ data is in N x ...
#         we want to keep num_samplexC of them.
#         if N > num_sample, we will randomly keep num_sample of them.
#         if N < num_sample, we will randomly duplicate samples.
#     """
#     N = data.shape[0]
#     if (N == num_sample):
#         return data, range(N)
#     elif (N > num_sample):
#         sample = np.random.choice(N, num_sample)
#         return data[sample, ...], sample
#     else:
#         sample = np.random.choice(N, num_sample-N)
#         dup_data = data[sample, ...]
#         return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)


def get_csv_list(path):
    # 获取csv文件列表
    csv_file_list = [[], [], []]
    file_list = os.listdir(path)
    for file_name in file_list:
        if file_name.endswith('csv'):
            filename = re.split('(\.)', file_name)[0]
            csv_type = len(filename) - len(str(int(filename)))
            csv_file_list[csv_type].append(path + file_name)
    return csv_file_list

def get_csv_data(path_list_arr):
    # 创建空的定维数组
    sum_data = np.empty([0, 1024, 3], dtype=np.float32)
    type_data = np.empty([0, 1], dtype=np.int32)
    # 类型序号
    type_serial = -1
    # 遍历每个类型的目录
    for path_list in path_list_arr:
        # 每个目录对应一种类型的数据
        type_serial += 1
        # 遍历每个csv文件
        for path in path_list:
            # 将每个csv文件读取为Numpy的数据
            data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
            # 计算空值补缺的数量
            empty_len = 1024 - len(data)
            # 完整的1024个元数据=csv文件数据+空值数据
            data_full = np.append(change_scale(data), np.empty([empty_len, 3], dtype=np.float32), axis=0)
            # 数据归并
            sum_data = np.append(sum_data, [data_full], axis=0)
            # 数据类型归并
            type_data = np.append(type_data, [[type_serial]], axis=0)
    return sum_data, type_data

if __name__ == "__main__":

    TRAIN_CSV_PATH = './pointdata2/traindata2/csv/'
    TEST_CSV_PATH = './pointdata2/testdata2/csv/'
    train_csv_list = get_csv_list(TRAIN_CSV_PATH)
    test_csv_list = get_csv_list(TEST_CSV_PATH)

    train_data, train_type_data = get_csv_data(train_csv_list)
    test_data, test_type_data = get_csv_data(test_csv_list)

    open("plant_train.h5", 'w')
    with h5py.File('plant_train.h5', 'r+') as f:
        f.create_dataset('data', data=train_data)
        f.create_dataset('faceId', data=np.empty([1024,1024], dtype=np.float32))
        f.create_dataset('label', data=train_type_data)
        f.create_dataset('normal', data=train_data)

    open("plant_test.h5", 'w')
    with h5py.File('plant_test.h5', 'r+') as f:
        f.create_dataset('data', data=test_data)
        f.create_dataset('faceId', data=np.empty([1024,1024], dtype=np.float32))
        f.create_dataset('label', data=test_type_data)
        f.create_dataset('normal', data=test_data)

    # DATA_FILES = getDataFiles(os.path.join(BASE_DIR, 'file_path.txt'))
    # num_sample = 4096
    # DATA_ALL = []
    # for fn in range(len(DATA_FILES)):
    #     current_data, current_label = loadDataFile(DATA_FILES[fn])
    #     change_data = change_scale(current_data)
    #     data_sample, index = sample_data(change_data, num_sample)
    #     data_label = np.hstack((data_sample, current_label[index]))
    #     DATA_ALL.append(data_label)

#     output = np.vstack(DATA_ALL)
#     output = output.reshape(-1, num_sample, 4)

#     # train and test number, save data
#     if not os.path.exists('plant_train.h5'):
#         with h5py.File('plant_train.h5') as f:
#             f.create_dataset('data', data=output[0:7, 0:3])
#             f['labels'] = output[0:8, 4]

#     if not os.path.exists('plant_test.h5'):
#         with h5py.File('plant_test.h5') as f:
#             f['data'] = output[7:9, 0:3]
#             f['labels'] = output[7:9, 4]
