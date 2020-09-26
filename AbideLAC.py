"""

ABIDE LAC.

Usage:
  nn.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  nn.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import utils.abide.prepare_utils as PrepareUtils

from docopt import docopt
from torch import nn

if __name__ == '__main__':
    # 开始计时
    start = time.process_time()

    arguments = docopt(__doc__)

    # 判断是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        gpu_status = True
    else:
        gpu_status = False

    # 表型数据位置
    pheno_path = './data/ABIDE/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    # 载入表型数据
    pheno = PrepareUtils.load_phenotypes(pheno_path)
    # 载入数据集
    hdf5 = PrepareUtils.hdf5_handler(bytes('./data/ABIDE/abide_lstm.hdf5', encoding='utf8'), 'a')

    # 脑图谱的选择
    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    # 保存所有数据
    dataset_x = []
    dataset_y = []

    # 超参数
    time_step = 10
    # 高斯的均值
    gaussian_mean = 0
    # 高斯的方差
    gaussian_sigma = 1
    # 下采样率
    down_sampling_rate = 3

    batch_size = 16
    # 训练周期
    EPOCHS = 50
    # 学习率
    learning_rate = 0.001
    # LSTM隐藏层数量
    lstm_hidden_num = 256
    # LSTM输出层数量
    lstm_output_num = 2
    # LSTM层数量
    lstm_layers_num = 2
    # 是否是双向LSTM
    bidirectional = True

    # 构建完整数据集
    hdf5_dataset = hdf5["patients"]
    for i in hdf5_dataset.keys():
        data_item_x = torch.from_numpy(np.array(hdf5["patients"][i]['cc200'], dtype=np.float32))
        data_item_y = hdf5["patients"][i].attrs["y"]
        dataset_x.append(data_item_x)
        dataset_y.append(data_item_y)

    # 把所有数据增加padding
    dataset_x = nn.utils.rnn.pad_sequence(dataset_x, batch_first=True, padding_value=0).numpy()
    dataset_y = np.array(dataset_y, dtype=np.long)

    # PM、GM、SM数据
    pm_sequence_ = []
    gm_sequence_ = []
    sm_sequence_ = []
    for data in dataset_x:
        # 原始数据
        raw_data = data.copy()
        # 高斯数据
        gaussian_data = raw_data.copy() + np.random.normal(gaussian_mean, gaussian_sigma, raw_data.shape)
        # 重采样数据
        resample_data = data.copy()

        # 获取时间序列长度
        time_length = raw_data.shape[0]
        pm_sequence_item_ = []
        gm_sequence_item_ = []
        sm_sequence_item_ = []

        # 计算PM、GM矩阵
        for i in range(time_length - time_step + 1):
            pm = raw_data[i: i + time_step, :]
            gm = gaussian_data[i: i + time_step, :]
            pm_sequence_item_.append(pm)
            gm_sequence_item_.append(gm)
        # 对数据进行下采样，计算SM矩阵
        sm_sequence_item_ = resample_data[::down_sampling_rate, :]

        # 保存PM、GM、SM列表
        pm_sequence_.append(pm_sequence_item_)
        gm_sequence_.append(gm_sequence_item_)
        sm_sequence_.append(sm_sequence_item_)

    # 获得PM、GM、SM数据
    pm_data = np.array(pm_sequence_)
    gm_data = np.array(gm_sequence_)
    sm_data = np.array(sm_sequence_)

    print('xxx')
