import torch
import torch.nn.modules as modules
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import utils.abide.prepare_utils as PrepareUtils
import numpy as np

from sklearn.model_selection import train_test_split
from model.TCNModel import TCN
from data.ABIDE.AbideData import AbideData


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCN Modeling - Abide')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval (default: 100')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--nhid', type=int, default=150,
                        help='number of hidden units per layer (default: 150)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')

    # 判断是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 解析参数
    args = parser.parse_args()
    # 设置Pytorch的随机种子
    torch.manual_seed(args.seed)

    # 输入序列的长度
    input_size = 88
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
    dataset_x = nn.utils.rnn.pad_sequence(dataset_x, batch_first=True, padding_value=0)
    dataset_y = torch.tensor(dataset_y, dtype=torch.long)
    # 把数据按照7:3比例进行拆分
    train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.3, shuffle=True)
    abideData_train = AbideData(train_x, train_y)
    abideData_test = AbideData(test_x, test_y)
    train_loader = DataLoader(dataset=abideData_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=abideData_test, batch_size=batch_size, shuffle=True)

    # 创建模型
    model = TCN(input_size, n_classes, n_channels, kernel_size=kernel_size, dropout=args.dropout).to(device)
    criterion = modules.CrossEntropyLoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)

    print('xxx')