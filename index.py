#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def get_csv_list(path):
    # 获取csv文件列表
    csv_file_list = [[],[],[]]
    file_list = os.listdir(path)
    for file_name in file_list:
        if file_name.endswith('csv'):
            filename = re.split('(\.)', file_name)[0]
            csv_type = len(filename) - len(str(int(filename)))
            csv_file_list[csv_type].append(path + file_name)
    return csv_file_list

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def loadDataFile(path):
    data = np.loadtxt(path)
    num = data.shape[0]
    point_xyz = data[:, 0:3]
    point_normal = data[:, 3:6]
    point_rgb = data[:, 6:9]
    # label just a example, should be repalced the real.
    # modlenet40 is 0-39, so the label can be 40 and 41
    label = np.ones((num, 1), dtype=int)+39
    return point_xyz, label


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


def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)


def get_csv_data(path_list_arr):
    type_serial=0
    count=0
    for path_list in path_list_arr:
        type_serial += 1
        print('这是第',type_serial,'类数据')
        sub_count=0
        for path in path_list:
            data_placeholder = np.empty([1024, 3], dtype=float)
            data = np.genfromtxt(path, delimiter=',',dtype=None)
            print(len(data))
            print(data_placeholder)
            sub_count+=1
            count+=1
            # print(data)
        print('小类总计：【',sub_count,'】')
    print('总计：【',count,'】')
    return []

if __name__ == "__main__":

    TRAIN_CSV_PATH = './pointdata2/traindata2/csv/'
    TEST_CSV_PATH = './pointdata2/testdata2/csv/'
    train_csv_list = get_csv_list(TRAIN_CSV_PATH)
    test_csv_list = get_csv_list(TEST_CSV_PATH)

    train_data = get_csv_data(train_csv_list)
    # test_data = get_csv_data(test_csv_list)

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
#             f['data'] = output[0:7, 0:3]
#             f['labels'] = output[0:8, 4]

#     if not os.path.exists('plant_test.h5'):
#         with h5py.File('plant_test.h5') as f:
#             f['data'] = output[7:9, 0:3]
#             f['labels'] = output[7:9, 4]
