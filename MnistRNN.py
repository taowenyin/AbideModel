import torch
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.modules as modules

from torch.utils.data import DataLoader
from model.RNNModel import RNNModel


# 重新打包Hidden和Cell
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == '__main__':
    # 判断是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    # 训练周期
    EPOCHS = 50
    # 学习率
    learning_rate = 0.001
    # 输入的特征数
    lstm_input_num = 28
    # LSTM隐藏层数量
    lstm_hidden_num = 128
    # LSTM输出层数量
    lstm_output_num = 10
    # LSTM层数量
    lstm_layers_num = 2

    # 下载训练和测试数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # 创建LSTM模型
    model = RNNModel(lstm_input_num, lstm_hidden_num, lstm_output_num, lstm_layers_num).to(device)
    criterion = modules.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开启训练
    model.train()
    total_step = len(train_loader)
    # 初始化Hidden和Cell
    (hidden, cell) = model.init_hidden_cell(batch_size)
    for epoch in range(EPOCHS):
        for i, (data_x, data_y) in enumerate(train_loader):
            data_x = data_x.view(-1, 28, 28).requires_grad_().to(device)
            data_y = data_y.to(device)

            (hidden, cell) = repackage_hidden((hidden, cell))
            optimizer.zero_grad()
            output, (hidden, cell) = model(data_x, hidden, cell)
            loss = criterion(output, data_y)
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, EPOCHS, i + 1, total_step, loss))
