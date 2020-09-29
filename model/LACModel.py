import torch
import torch.nn.modules as modules


class LACModel(modules.Module):
    def __init__(self, input_size, hidden_size, kernel_size, out_channels, output_size, num_layers=1, dropout=0, bidirectional=False):
        super(LACModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # 创建LSTM
        self.rnn = modules.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        # LSTM激活函数
        self.rnn_act = modules.ReLU()
        # 创建1D-CNN
        self.cnn = modules.Conv1d(1, self.out_channels, self.kernel_size)
        # 1D-CNN激活函数
        self.cnn_act = modules.Tanh()
        # Dropout层
        self.drop = modules.Dropout(dropout)

    # 初始化Hidden和Cell
    def init_hidden_cell(self, batch_size, bidirectional=False):
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # 获取权重对象
        weight = next(self.parameters())
        # 初始化权重
        hidden = weight.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
        cell = weight.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size)

        return hidden, cell

    def forward(self, data_x, hidden, cell):
        output, (hidden, cell) = self.rnn(data_x, (hidden, cell))
        # 重新组织数据
        hidden = hidden.permute(1, 0, 2)

        output = self.cnn(hidden)

        return output, (hidden, cell)

