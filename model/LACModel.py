import torch.nn.modules as modules

from model.LACModelUnit import LACModelUnit


class LACMode(modules.Module):
    def __init__(self, input_size, hidden_size, kernel_size, out_channels, output_size, num_layers=1,
                 dropout=0, bidirectional=False):
        super(LACMode, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.pm_model = LACModelUnit(input_size, hidden_size, kernel_size, out_channels, output_size,
                                     num_layers, dropout, bidirectional)
        self.gm_model = LACModelUnit(input_size, hidden_size, kernel_size, out_channels, output_size,
                                     num_layers, dropout, bidirectional)
        self.sm_model = LACModelUnit(input_size, hidden_size, kernel_size, out_channels, output_size,
                                     num_layers, dropout, bidirectional)

    # 初始化Hidden和Cell
    def init_hidden_cell(self, batch_size):
        if self.bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # 获取权重对象
        weight = next(self.parameters())
        # 初始化权重
        hidden = weight.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
        cell = weight.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size)

        return hidden, cell

    def forward(self, pm_x, gm_x, sm_x, pm_hidden, pm_cell, gm_hidden, gm_cell, sm_hidden, sm_cell):
        pm_output, (pm_hidden, pm_cell) = self.pm_model(pm_x, pm_hidden, pm_cell)
        gm_output, (gm_hidden, gm_cell) = self.gm_model(gm_x, gm_hidden, gm_cell)
        sm_output, (sm_hidden, sm_cell) = self.sm_model(sm_x, sm_hidden, sm_cell)

        return pm_output, (pm_hidden, pm_cell), gm_output, (gm_hidden, gm_cell), sm_output, (sm_hidden, sm_cell)