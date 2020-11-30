import torch
import torch.nn.modules as modules
import torch.nn.functional as F

from torch.nn.utils import weight_norm


# 定义剪裁模块用于减去多余的padding
class Chomp1d(modules.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 定义残差模块
class TemporalBlock(modules.Module):
    """
    相当于一个Residual block

    输入数据形状：(N, C, L)，N表示batch_size，C表示通道数即数据维度，L表示数据长度
    n_inputs: 输入通道
    n_outputs: 输出通道
    kernel_size: 卷积核大小
    stride: 卷积的步长
    dilation: 扩张系数
    padding: 边缘填充的参数
    dropout: dropout系数
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 定义残差模块的第一层扩张卷积
        # 经过conv，输出的size为(Batch, input_channel, seq_len + padding)，并归一化模型的参数
        self.conv1 = weight_norm(modules.Conv1d(n_inputs, n_outputs, kernel_size,
                                                stride=stride, padding=padding, dilation=dilation))
        # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.chomp1 = Chomp1d(padding)
        self.relu1 = modules.ReLU()
        self.dropout1 = modules.Dropout(dropout)

        #定义残差模块的第二层扩张卷积
        self.conv2 = weight_norm(modules.Conv1d(n_outputs, n_outputs, kernel_size,
                                                stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = modules.ReLU()
        self.dropout2 = modules.Dropout(dropout)

        # 将卷积模块进行串联构成序列
        self.net = modules.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                      self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 如果输入通道和输出通道不相同，那么通过1x1卷积进行降维，保持通道相同
        self.downsample = modules.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = modules.ReLU()
        self.init_weights()

    def init_weights(self):
        # 初始化权重为均值为0，标准差为0.01的正态分布
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # 如果降维，那么那么降维层的权重也要进行初始化
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # 残差模块
        return self.relu(out + res)


# 定义TCN架构
class TemporalConvNet(modules.Module):
    """
    num_inputs: 输入通道数
    num_channels: 每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # 扩张系数随着网络层级的增加而呈现指数级增加，以此来增大感受野
            dilation_size = 2 ** i
            # 计算输入通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            # 确定每一层的输出通道数
            out_channels = num_channels[i]
            # 从num_channels中抽取每个残差模块的输入通道数和输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.network = modules.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


# TCN模型
class TCN(modules.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = modules.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)
