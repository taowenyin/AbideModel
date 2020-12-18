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
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
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
parser.add_argument('--log_interval', type=int, default=16, metavar='N',
                    help='report interval (default: 16')
# parser.add_argument('--log_interval', type=int, default=4, metavar='N',
#                     help='report interval (default: 4')
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
dataset_y = torch.tensor(dataset_y, dtype=torch.long)
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
criterion = modules.NLLLoss()
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


# 训练函数
def train(ep):
    train_pred = []
    train_interval_loss = 0
    train_loss = 0
    count = 0
    model.train()
    # 获得训练数据的索引
    train_idx_list = np.arange(len(train_x), dtype=np.int32)
    # 打乱索引
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
    # for idx, (data_x, data_y) in enumerate(train_loader):
        # 获取训练数据
        data = train_x[idx].transpose(0, 1).unsqueeze(0).requires_grad_().to(device)
        target = torch.tensor([train_y[idx]], device=device)
        # data = data_x.transpose(1, 2).requires_grad_().to(device)
        # target = data_y.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_interval_loss += loss
        train_loss += loss
        count += output.size(0)

        # 添加训练预测结果
        train_pred.extend(output.data.max(1, keepdim=True)[1].view(1, output.size(0))[0].cpu().numpy().tolist())

        if idx > 0 and idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep,
                count,
                len(train_x),
                100. * count / len(train_x),
                train_interval_loss.item() / args.log_interval))
            train_interval_loss = 0

    return train_loss / len(train_x), train_pred


# 测试函数
def test():
    model.eval()
    test_pred = []
    total_loss = 0
    count = 0
    correct = 0
    # 获得测试数据的索引
    test_idx_list = np.arange(len(test_x), dtype=np.int32)
    with torch.no_grad():
        for idx in test_idx_list:
        # for idx, (data_x, data_y) in enumerate(test_loader):
            # 获取测试数据
            data = test_x[idx].transpose(0, 1).unsqueeze(0).requires_grad_().to(device)
            target = torch.tensor([test_y[idx]], device=device)
            # data = data_x.transpose(1, 2).requires_grad_().to(device)
            # target = data_y.to(device)

            output = model(data)
            total_loss += criterion(output, target)
            count += output.size(0)
            # 获取结果中最大值的索引
            pred = output.data.max(1, keepdim=True)[1]
            # 把预测结果与标签进行形状统一，并判断是否相同
            correct += pred.eq(target.data.view_as(pred)).sum()

            # 添加训练预测结果
            test_pred.extend(pred.view(1, output.size(0))[0].cpu().numpy().tolist())

        test_loss = total_loss / count
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,
            correct,
            len(test_x),
            100. * correct / len(test_x)))

        return test_loss, test_pred


if __name__ == '__main__':
    plt.figure(figsize=(16, 9))

    # 保存训练损失
    train_loss_data = []
    # 准确率
    train_accuracy_data = []
    # 召回率
    train_recall_data = []
    # F1值
    train_f1_data = []

    # 保存测试损失
    test_loss_data = []
    # 准确率
    test_accuracy_data = []
    # 召回率
    test_recall_data = []
    # F1值
    test_f1_data = []

    for epoch in range(1, epochs + 1):
        train_loss, train_pred = train(epoch)
        train_loss_data.append(train_loss)

        test_loss, test_pred = test()
        test_loss_data.append(test_loss)

        # 保存训练的各类评价分数
        train_accuracy_data.append(accuracy_score(train_y, train_pred))
        train_recall_data.append(recall_score(train_y, train_pred))
        train_f1_data.append(f1_score(train_y, train_pred))

        # 保存测试的各类评价分数
        test_accuracy_data.append(accuracy_score(test_y, test_pred))
        test_recall_data.append(recall_score(test_y, test_pred))
        test_f1_data.append(f1_score(test_y, test_pred))

        # 动态修改学习率
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    plt.subplot(2, 4, 1)
    plt.plot(np.arange(epochs) + 1, train_accuracy_data, label='Train Accuracy')
    plt.xlabel('EPOCH')
    plt.title('Train Accuracy')
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.plot(np.arange(epochs) + 1, train_recall_data, label='Train Recall')
    plt.xlabel('EPOCH')
    plt.title('Train Recall')
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.plot(np.arange(epochs) + 1, train_f1_data, label='Train F1')
    plt.xlabel('EPOCH')
    plt.title('Train F1')
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.plot(np.arange(len(train_loss_data)) + 1, train_loss_data, label='Train Loss')
    plt.xlabel('EPOCH')
    plt.title('Train Loss')
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.plot(np.arange(epochs) + 1, test_accuracy_data, label='Test Accuracy')
    plt.xlabel('EPOCH')
    plt.title('Test Accuracy')
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.plot(np.arange(epochs) + 1, test_recall_data, label='Test Recall')
    plt.xlabel('EPOCH')
    plt.title('Test Recall')
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.plot(np.arange(epochs) + 1, test_f1_data, label='Test F1')
    plt.xlabel('EPOCH')
    plt.title('Test F1')
    plt.legend()

    plt.subplot(2, 4, 8)
    plt.plot(np.arange(len(test_loss_data)) + 1, test_loss_data, label='Test Loss')
    plt.xlabel('EPOCH')
    plt.title('Test Loss')
    plt.legend()

    # plt.suptitle('Batch Train')
    plt.suptitle('One Train')

    plt.show()
