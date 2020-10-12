import torch
import torch.nn.modules as modules

from model.LACModelUnit import LACModelUnit
from torch import nn


class LACModel(modules.Module):
    def __init__(self, model_name, input_size, hidden_size, batch_size, kernel_size, out_channels,
                 output_size, num_layers=1, dropout=0, bidirectional=False, bn=False):
        super(LACModel, self).__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.bn = bn

        self.pm_model = LACModelUnit('PM-{0}'.format(model_name), input_size, hidden_size, batch_size,
                                     kernel_size, out_channels, num_layers, dropout, bidirectional, bn)
        self.gm_model = LACModelUnit('GM-{0}'.format(model_name), input_size, hidden_size, batch_size,
                                     kernel_size, out_channels, num_layers, dropout, bidirectional, bn)
        self.sm_model = LACModelUnit('SM-{0}'.format(model_name), input_size, hidden_size, batch_size,
                                     kernel_size, out_channels, num_layers, dropout, bidirectional, bn)
        # 判断是否是双线LSTM
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # 创建全连接层
        self.fc = modules.Linear((hidden_size * num_directions - kernel_size + 1) * out_channels * 3, output_size)
        # BN层
        self.bn = modules.BatchNorm1d(output_size)
        # LSTM激活函数
        self.activation = modules.Softmax(dim=1)

    def forward(self, pm_x, gm_x, sm_x):
        # 获得PM、GM、SM数据拉伸后的一维数据
        pm_output = self.pm_model(pm_x)
        gm_output = self.gm_model(gm_x)
        sm_output = self.sm_model(sm_x)

        # 合并PM、GM、SM
        output = torch.cat((pm_output, gm_output, sm_output), dim=1)

        output = self.fc(output)
        if self.bn:
            output = self.bn(output)
        output = self.activation(output)

        return output
