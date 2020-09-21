import torch
import torch.nn.modules as modules


class RNNModel(modules.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # 创建模型
        self.rnn = modules.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=self.dropout)
        if self.dropout != 0:
            self.drop = modules.Dropout(dropout)
        # 创建全连接层
        self.fc = modules.Linear(hidden_size, output_size)
        # 设置激活函数
        self.activation = modules.LogSoftmax(dim=1)

    # 初始化Hidden和Cell
    def init_hidden_cell(self, batch_size):
        # 获取权重对象
        weight = next(self.parameters())
        # 初始化权重
        hidden = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)
        cell = weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

        return hidden, cell

    def forward(self, input_x, hidden, cell):
        output, (hidden, cell) = self.rnn(input_x, (hidden, cell))
        if self.dropout != 0:
            output = self.drop(output)
        output = self.fc(output)
        # 提取最后一个时间节点的数据
        output = output[:, -1, :]
        output = self.activation(output)
        return output, (hidden, cell)
