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
import torch.nn.modules as modules
import time
import utils.abide.prepare_utils as PrepareUtils
import utils.functions as functions

from docopt import docopt
from torch import nn
from sklearn.model_selection import train_test_split
from data.ABIDE.AbideLacData import AbideLacData
from torch.utils.data import DataLoader
from model.LACModelUnit import LACModelUnit
from model.LACModel import LACMode

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
    # 1D卷积核大小
    kernel_size = 3
    # 1D卷积的输出通道数
    out_channels = 4
    # LSTM隐藏层数量
    lstm_hidden_num = 256
    # LSTM输出层数量
    output_size = 2
    # LSTM层数量
    lstm_layers_num = 1
    # 是否是双向LSTM
    bidirectional = False
    # dropout大小
    dropout = 0.2

    # 每批数据的大小
    batch_size = 16
    # 训练周期
    EPOCHS = 50
    # 学习率
    learning_rate = 0.001
    # 模型序列数量
    model_sequence_size = 9

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
        gaussian_data = raw_data.copy() + np.array(
            np.random.normal(gaussian_mean, gaussian_sigma, raw_data.shape), dtype=np.float32)
        # 对数据进行下采样
        resample_data = data.copy()[::down_sampling_rate, :]

        # 获取时间序列长度
        raw_length = raw_data.shape[0]
        resample_length = resample_data.shape[0]
        pm_sequence_item_ = []
        gm_sequence_item_ = []
        sm_sequence_item_ = []

        # 计算PM、GM矩阵
        for i in range(raw_length - time_step + 1):
            pm = raw_data[i: i + time_step, :].flatten()
            gm = gaussian_data[i: i + time_step, :].flatten()
            pm_sequence_item_.append(pm)
            gm_sequence_item_.append(gm)
        # 计算SM矩阵
        for i in range(resample_length - time_step + 1):
            sm = resample_data[i: i + time_step, :].flatten()
            sm_sequence_item_.append(sm)

        # 保存PM、GM、SM列表
        pm_sequence_.append(pm_sequence_item_)
        gm_sequence_.append(gm_sequence_item_)
        sm_sequence_.append(sm_sequence_item_)
    # 获得PM、GM、SM数据
    pm_data = np.array(pm_sequence_)
    gm_data = np.array(gm_sequence_)
    sm_data = np.array(sm_sequence_)

    lac_data = AbideLacData(pm_data, gm_data, sm_data, dataset_y)
    lac_loader = DataLoader(dataset=lac_data, batch_size=batch_size, shuffle=True)

    model_sequence = []
    hidden_cell_sequence = []
    # 创建LSTM模型
    for i in range(model_sequence_size):
        # 初始化模型
        model = LACMode(lac_data.get_feature_size(), lstm_hidden_num, kernel_size, out_channels,
                        output_size, num_layers=lstm_layers_num, dropout=dropout,
                        bidirectional=bidirectional).to(device)
        # 初始化PM、GM、SM的Hidden和Cell
        hidden_cell_sequence_item = [model.init_hidden_cell(batch_size),
                                     model.init_hidden_cell(batch_size),
                                     model.init_hidden_cell(batch_size)]
        model_sequence.append(model)
        hidden_cell_sequence.append(hidden_cell_sequence_item)
    # 初始化优化器
    optimizer = torch.optim.Adam([{"params": model.parameters()} for model in model_sequence], lr=learning_rate)
    # 初始化损失函数
    criterion = modules.CrossEntropyLoss()

    # 开启训练
    for i in range(model_sequence_size):
        model = model_sequence[i]
        model.train()

    total_step = len(lac_loader)
    for epoch in range(EPOCHS):
        for i, data in enumerate(lac_loader):
            # 保存模型结果
            model_result = []
            # 投票结果
            result = []

            pm_x = data[0].requires_grad_().to(device)
            gm_x = data[1].requires_grad_().to(device)
            sm_x = data[2].requires_grad_().to(device)
            data_y = data[3].to(device)

            # 清空所有网络的梯度
            optimizer.zero_grad()
            for j in range(model_sequence_size):
                # 获取模型
                model = model_sequence[j]

                # 解包Hidden和Cell
                (pm_hidden, pm_cell) = functions.repackage_hidden(hidden_cell_sequence[j][0])
                (gm_hidden, gm_cell) = functions.repackage_hidden(hidden_cell_sequence[j][1])
                (sm_hidden, sm_cell) = functions.repackage_hidden(hidden_cell_sequence[j][2])

                pm_hidden_ = pm_cell_ = gm_hidden_ = gm_cell_ = sm_hidden_ = sm_cell_ = None
                # 获取数据形状
                curr_batch_size = pm_x.shape[0]
                if curr_batch_size < batch_size:
                    # 参数备份
                    pm_hidden_ = pm_hidden.clone()
                    pm_cell_ = pm_cell.clone()
                    gm_hidden_ = gm_hidden.clone()
                    gm_cell_ = gm_cell.clone()
                    sm_hidden_ = sm_hidden.clone()
                    sm_cell_ = sm_cell.clone()

                    # 切换部分数据
                    pm_hidden = pm_hidden[:, 0:curr_batch_size, :]
                    pm_cell = pm_cell[:, 0:curr_batch_size, :]
                    gm_hidden = gm_hidden[:, 0:curr_batch_size, :]
                    gm_cell = gm_cell[:, 0:curr_batch_size, :]
                    sm_hidden = sm_hidden[:, 0:curr_batch_size, :]
                    sm_cell = sm_cell[:, 0:curr_batch_size, :]

                # 模型计算
                output, (pm_hidden, pm_cell), (gm_hidden, gm_cell), (sm_hidden, sm_cell) = model(
                    pm_x, gm_x, sm_x, pm_hidden, pm_cell, gm_hidden, gm_cell, sm_hidden, sm_cell)
                loss = criterion(output, data_y)
                loss.backward()

                print('{} EPOCH {} data batch {} model loss:{:.4f}'.format(epoch, i, j, loss))

                # 恢复参数数据
                if curr_batch_size < batch_size:
                    pm_hidden_[:, 0:curr_batch_size, :] = pm_hidden
                    pm_cell_[:, 0:curr_batch_size, :] = pm_cell
                    gm_hidden_[:, 0:curr_batch_size, :] = gm_hidden
                    gm_cell_[:, 0:curr_batch_size, :] = gm_cell
                    sm_hidden_[:, 0:curr_batch_size, :] = sm_hidden
                    sm_cell_[:, 0:curr_batch_size, :] = sm_cell

                    pm_hidden = pm_hidden_
                    pm_cell = pm_cell_
                    gm_hidden = gm_hidden_
                    gm_cell = gm_cell_
                    sm_hidden = sm_hidden_
                    sm_cell = sm_cell_

                # 更新Hidden和Cell
                hidden_cell_sequence[j][0] = (pm_hidden, pm_cell)
                hidden_cell_sequence[j][1] = (gm_hidden, gm_cell)
                hidden_cell_sequence[j][2] = (sm_hidden, sm_cell)
                # 获得每个模型的结果
                model_result.append(output)
            # 优化参数
            optimizer.step()

            # ===================测试时进行投票===================
            # # 多模型结果重新布局
            # shape = model_result[0].shape
            # model_result = torch.cat(model_result, dim=0).view(-1, shape[0], shape[1])
            # # 对多模型结果进行投票解决
            # for k in range(model_result.shape[1]):
            #     vote = model_result[:, k, :]
            #     vote = torch.argmax(vote, dim=1)
            #     negative = vote[vote == 0].numel()
            #     positive = vote[vote == 1].numel()
            #
            #     # 计算投票结果
            #     if negative > positive:
            #         result.append(0)
            #     else:
            #         result.append(1)
            print('===One Data Batch Over===')

    print('xxx')
