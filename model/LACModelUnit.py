import torch
import torch.nn.modules as modules
import numpy as np
import utils.functions as functions


class LACModelUnit(modules.Module):
    def __init__(self, model_name, input_size, hidden_size, batch_size, kernel_size,
                 out_channels, num_layers=1, dropout=0, bidirectional=False):
        super(LACModelUnit, self).__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # 创建LSTM
        self.rnn = modules.LSTM(self.input_size, self.hidden_size, self.num_layers,
                                batch_first=True, bidirectional=self.bidirectional)
        # LSTM激活函数
        self.rnn_act = modules.ReLU()
        # 创建1D-CNN
        self.cnn = modules.Conv1d(1, self.out_channels, self.kernel_size)
        # BN层
        self.bn = modules.BatchNorm1d(self.out_channels)
        # 1D-CNN激活函数
        self.cnn_act = modules.Tanh()
        # Dropout层
        self.drop = modules.Dropout(dropout)
        # 获得GPU数量
        self.cuda_ids = np.arange(torch.cuda.device_count())
        # 判断是否使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化LSTM参数
        self.lstm_hidden, self.lstm_cell = self.init_hidden_cell(self.batch_size / len(self.cuda_ids))

    # 初始化Hidden和Cell
    def init_hidden_cell(self, batch_size):
        if self.bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # 获取权重对象
        weight = next(self.parameters())
        # 初始化权重
        hidden = weight.new_zeros(self.num_layers * num_directions, int(batch_size), self.hidden_size).to(self.device)
        cell = weight.new_zeros(self.num_layers * num_directions, int(batch_size), self.hidden_size).to(self.device)

        return hidden, cell

    def forward(self, data_x):
        output, (self.lstm_hidden, self.lstm_cell) = self.rnn(data_x, (self.lstm_hidden, self.lstm_cell))
        # 解绑Hidden和Cell
        (self.lstm_hidden, self.lstm_cell) = functions.repackage_hidden((self.lstm_hidden, self.lstm_cell))

        # 经过ReLU函数激活
        output = self.rnn_act(self.lstm_hidden)
        # 重新组织数据
        output = output.permute(1, 0, 2)
        output = output.reshape(output.shape[0], 1, -1)

        # output size: batch_siz * out_channels * (lstm_hidden_num * bidirectional - kernel_size + 1)
        output = self.cnn(output)
        # 经过BN层
        # output = self.bn(output)
        # 经过Tanh函数激活
        output = self.cnn_act(output)
        # 把二维数据拉为一维数据
        output = output.reshape(output.shape[0], -1)

        # print('{0} forward'.format(self.model_name))

        return output
