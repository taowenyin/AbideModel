import torch

from torch import nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 创建模型
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 创建全连接层
        self.decoder = nn.Linear(hidden_size, output_size)
        # 设置激活函数
        self.softmax = nn.LogSoftmax(dim=1)

        if torch.cuda.is_available():
            self.gpu_status = True
        else:
            self.gpu_status = False

    # 初始化Hidden和Cell
    def init_hidden_cell(self, batch_size):
        weight = next(self.parameters())
        hidden = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        cell = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

        if self.gpu_status:
            hidden = hidden.cuda()
            cell = cell.cuda()

        return hidden, cell

    def forward(self, input_x, hidden, cell):
        output, (hidden, cell) = self.lstm(input_x, (hidden, cell))
        # output, output_length = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # 提取最后一个时间节点的数据
        output = output[:, -1, :]
        output = self.decoder(output)
        output = self.softmax(output)
        return output, (hidden, cell)