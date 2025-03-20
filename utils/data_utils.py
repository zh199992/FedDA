import numpy as np
import os
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)

    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]


# def read_data(dataset, idx, args, is_train=True):
#     if is_train:
#         train_data_dir = os.path.join('../data', dataset, 'processed',args.dp)
#
#         train_feature = train_data_dir + "train_FD00" +str(idx)+"feature"+ str(args.window_size)+'.pt'
#         train_label = train_data_dir + "train_FD00" +str(idx)+"label"+ str(args.window_size)+'.pt'
#         train_set = MyDataset(train_feature, train_label)
#
#         return train_set
#
#     else:
#         test_data_dir = os.path.join('../data', dataset, 'processed',args.dp)
#         test_feature = test_data_dir + "test_FD00" +str(idx)+str(args.window_size)+'.pt'
#         test_label = test_data_dir + "RUL_FD00" +str(idx)+ str(args.window_size)+'.pt'
#         test_set = MyDataset(test_feature, test_label)
#
#         return test_set
def read_client_data(dataset, idx, args,  is_train=True):  #调用read_data   输出元组列表
    current_directory = os.getcwd()
    if is_train:
        train_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', dataset, 'processed',args.dp)+'/'

        train_feature = torch.load(train_data_dir + "train_FD00" +str(idx)+"feature"+ str(args.window_size)+'.pt')
        train_label = torch.load(train_data_dir + "train_FD00" +str(idx)+"label"+ str(args.window_size)+'.pt')
        train_set = MyDataset(train_feature, train_label)

        return train_set

    else:
        test_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', dataset, 'processed',args.dp)+'/'
        test_feature = torch.load(test_data_dir + "RUL_FD00" +str(idx)+str(args.window_size)+'.pt')
        test_label = torch.load(test_data_dir + "test_FD00" +str(idx)+ str(args.window_size)+'.pt')
        test_set = MyDataset(test_feature, test_label)

        return test_set


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = [len(d) for d in datasets]
        self.total_size = sum(self.cumulative_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        for i, size in enumerate(self.cumulative_sizes):
            if idx < size:
                data, _ = self.datasets[i][idx]
                return data, i  # 返回数据和域索引
            idx -= size