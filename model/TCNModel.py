import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# 定义因果卷积
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 对继承自父类的属性进行初始化
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 定义残差模块
class TemporalBlock(nn.Module):
    """
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
        # 归一化模型的参数
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 定义残差模块的第一层扩张卷积
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        #定义残差模块的第二层扩张卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 将卷积模块进行串联构成序列
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        # 如果输入通道和输出通道不相同，那么通过1x1卷积进行降维，保持通道相同
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # 初始化权重为均值为0，标准差为0.01的正态分布
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # 如果降维，那么那么降维层的权重也要进行初始化
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # 残差模块
        return self.relu(out + res)


# 定义TCN架构
class TemporalConvNet(nn.Module):
    """
    num_inputs:
    num_channels: 各层卷积运算的输出通道数或卷积核的数量，该长度即需要执行的卷基层数量
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # 扩张系数随着网络层级的增加而呈现指数级增加，以此来增大感受野
            dilation_size = 2 ** i
            # 计算输入输出通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # 从num_channels中抽取每个残差模块的输入通道数和输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # 将所有残差模块堆叠起来组成一个深度卷积网络
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
