import torch
import torch.nn.modules as modules


class LACModelUnit(modules.Module):
    def __init__(self, input_size, hidden_size, kernel_size,
                 out_channels, num_layers=1, dropout=0, bidirectional=False):
        super(LACModelUnit, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
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
        # 1D-CNN激活函数
        self.cnn_act = modules.Tanh()
        # BN层
        self.bn = modules.BatchNorm1d(self.out_channels)
        # Dropout层
        self.drop = modules.Dropout(dropout)

    def forward(self, data_x, hidden, cell):
        output, (hidden, cell) = self.rnn(data_x, (hidden, cell))
        # 经过ReLU函数激活
        output = self.rnn_act(hidden)
        # 重新组织数据
        output = output.permute(1, 0, 2)
        output = output.reshape(output.shape[0], 1, -1)

        # output size: batch_siz * out_channels * (lstm_hidden_num * bidirectional - kernel_size + 1)
        output = self.cnn(output)
        # 经过Tanh函数激活
        output = self.cnn_act(output)
        # 经过BN层
        output = self.bn(output)
        # 把二维数据拉为一维数据
        output = output.reshape(output.shape[0], -1)

        return output, (hidden, cell)

