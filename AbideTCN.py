import torch
import torch.nn.modules as modules
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import utils.abide.prepare_utils as PrepareUtils
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from model.TCNModel import TCN
from data.ABIDE.AbideData import AbideData


"""
模型运行时的参数
"""
parser = argparse.ArgumentParser(description='TCN Modeling - Abide')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log_interval', type=int, default=40, metavar='N',
                    help='report interval (default: 40')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

"""
Pytorch配置
"""
# 判断是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 解析参数
args = parser.parse_args()
# 设置Pytorch的随机种子
torch.manual_seed(args.seed)

"""
模型的各类参数
"""
# 输入序列的通道数
input_size = 200
# 设置每一层的通道数
n_channels = [args.nhid] * args.levels
# 卷积核大小
kernel_size = args.ksize
# dropout大小
dropout = args.dropout
# 分类的种类
n_classes = 2
# 每批的大小
batch_size = args.batch_size
# 训练的次数
epochs = args.epochs
# 模型的学习率
lr = args.lr
# 序列长度
seq_length = 316

"""
构建模型的数据
"""
# 载入数据集
hdf5 = PrepareUtils.hdf5_handler(bytes('./data/ABIDE/abide_lstm.hdf5', encoding='utf8'), 'a')
# 保存所有数据
dataset_x = []
dataset_y = []
# 构建完整数据集
hdf5_dataset = hdf5["patients"]
for i in hdf5_dataset.keys():
    data_item_x = torch.from_numpy(np.array(hdf5["patients"][i]['cc200'], dtype=np.float32))
    data_item_y = hdf5["patients"][i].attrs["y"]
    dataset_x.append(data_item_x)
    dataset_y.append(data_item_y)
# 把所有数据增加padding
# dataset_x = nn.utils.rnn.pad_sequence(dataset_x, batch_first=True, padding_value=0)
# dataset_y = torch.tensor(dataset_y, dtype=torch.long)
# 把数据按照7:3比例进行拆分
train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.3, shuffle=True)
# abideData_train = AbideData(train_x, train_y)
# abideData_test = AbideData(test_x, test_y)
# train_loader = DataLoader(dataset=abideData_train, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(dataset=abideData_test, batch_size=batch_size, shuffle=True)

"""
创建模型
"""
model = TCN(input_size, n_classes, n_channels, kernel_size=kernel_size, dropout=args.dropout).to(device)
criterion = modules.CrossEntropyLoss()
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


# 训练函数
def train(ep):
    train_interval_loss = 0
    train_loss = 0
    count = 0
    model.train()
    # 获得训练数据的索引
    train_idx_list = np.arange(len(train_x), dtype=np.int32)
    # 打乱索引
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        # 获取训练数据
        data = train_x[idx].transpose(0, 1).requires_grad_().to(device)
        target = torch.tensor([train_y[idx]], device=device)
        optimizer.zero_grad()
        output = model(data.unsqueeze(0))
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_interval_loss += loss
        train_loss += loss
        count += output.size(0)

        if idx > 0 and idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep,
                count,
                len(train_x),
                100. * count / len(train_x),
                train_interval_loss.item() / args.log_interval))
            train_interval_loss = 0

    return train_loss / len(train_x)


# 测试函数
def test():
    model.eval()
    total_loss = 0
    count = 0
    correct = 0
    # 获得测试数据的索引
    test_idx_list = np.arange(len(test_x), dtype=np.int32)
    with torch.no_grad():
        for idx in test_idx_list:
            # 获取测试数据
            data = test_x[idx].transpose(0, 1).requires_grad_().to(device)
            target = torch.tensor([test_y[idx]], device=device)
            output = model(data.unsqueeze(0))
            total_loss += F.nll_loss(output, target)
            count += output.size(0)
            # 获取结果中最大值的索引
            pred = output.data.max(1, keepdim=True)[1]
            # 把预测结果与标签进行形状统一，并判断是否相同
            a = pred.eq(target.data.view_as(pred))
            correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss = total_loss / count
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,
            correct,
            len(test_x),
            100. * correct / len(test_x)))

        return test_loss

if __name__ == '__main__':
    # 保存训练损失
    train_loss = []
    # 保存测试损失
    test_loss = []
    for epoch in range(1, epochs + 1):
        train_loss.append(train(epoch))
        test_loss.append(test())
        # 动态修改学习率
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    plt.plot(np.arange(len(train_loss)) + 1, train_loss, label='Train Loss')
    plt.plot(np.arange(len(test_loss)) + 1, test_loss, label='Test Loss')
    plt.xlabel('EPOCH')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
