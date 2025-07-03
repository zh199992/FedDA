import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import resample


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

def read_client_data_iid(dataset, split_sizes, start, idx, args,  is_train=True):  #调用read_data   输出元组列表
    current_directory = os.getcwd()
    if is_train:
        train_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', dataset, 'processed',args.dp)+'/'

        # train_feature = torch.load(train_data_dir + "train_FD00" +str(idx)+"feature"+ str(args.window_size)+'.pt')
        # train_label = torch.load(train_data_dir + "train_FD00" +str(idx)+"label"+ str(args.window_size)+'.pt')
        train_feature = torch.load(train_data_dir + "train_FD001" +"feature"+ str(args.window_size)+'.pt')[start:start+split_sizes[idx-1]]
        train_label = torch.load(train_data_dir + "train_FD001" +"label"+ str(args.window_size)+'.pt')[start:start+split_sizes[idx-1]]
        train_set = MyDataset(train_feature, train_label)

        return train_set

    else:#应该整个还是四分之一？ 两个都用吧
        test_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', dataset, 'processed',args.dp)+'/'
        # test_feature = torch.load(test_data_dir + "RUL_FD00" +str(idx)+str(args.window_size)+'.pt')
        # test_label = torch.load(test_data_dir + "test_FD00" +str(idx)+ str(args.window_size)+'.pt')
        test_feature = torch.load(test_data_dir + "RUL_FD001" + str(args.window_size) + '.pt')[start:start+split_sizes[idx-1]]
        test_label = torch.load(test_data_dir + "test_FD001" + str(args.window_size) + '.pt')[start:start+split_sizes[idx-1]]
        test_set = MyDataset(test_feature, test_label)

        return test_set

def read_client_data_centralized(dataset, args,  is_train=True):  #调用read_data   输出元组列表
    current_directory = os.getcwd()
    if is_train:
        train_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', dataset, 'processed',args.dp)+'/'
        train_feature=[]
        train_label=[]
        for i in range(1,5):
            train_feature.append(torch.load(train_data_dir + "train_FD00" +str(i)+"feature"+ str(args.window_size)+'.pt'))
            train_label.append(torch.load(train_data_dir + "train_FD00" +str(i)+"label"+ str(args.window_size)+'.pt'))

        train_feature=torch.cat(train_feature,dim=0)
        train_label=torch.cat(train_label,dim=0)
        train_set = MyDataset(train_feature, train_label)

        return train_set

    else:#应该整个还是四分之一？ 两个都用吧
        test_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', dataset, 'processed',args.dp)+'/'
        test_feature = []
        test_label= []
        for i in range(1,5):
            test_feature.append(torch.load(test_data_dir + "RUL_FD00" +str(i)+str(args.window_size)+'.pt'))
            test_label.append(torch.load(test_data_dir + "test_FD00" +str(i)+ str(args.window_size)+'.pt'))

        test_feature=torch.cat(test_feature,dim=0)
        test_label=torch.cat(test_label,dim=0)
        test_set = MyDataset(test_feature, test_label)

        return test_set
def calculate_split_sizes(dataset, args, num_splits=4):
    data_dir = os.path.join('/home/zhouheng/project/FedDA/data', dataset, 'processed', args.dp) + '/'
    train_feature = torch.load(data_dir + "train_FD001" + "feature" + str(args.window_size) + '.pt')
    total=train_feature.size(0)
    base = total // num_splits
    remainder = total % num_splits
    split_sizes_train = [base + 1 if i < remainder else base for i in range(num_splits)]

    test_feature = torch.load(data_dir + "RUL_FD001" + str(args.window_size) + '.pt')
    total=test_feature.size(0)
    base = total // num_splits
    remainder = total % num_splits
    split_sizes_test = [base + 1 if i < remainder else base for i in range(num_splits)]
    return split_sizes_train, split_sizes_test

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

def visualize(features_list,fig_name, axes):#是不是应该存下来？  可以把每一轮的都保存吗？ 我要看client各自提取的特征还是聚合后的模型提取的特征？
    flat=nn.Flatten()
    features1_np = flat(features_list[0]).cpu().detach().numpy()
    features2_np = flat(features_list[1]).cpu().detach().numpy()
    features3_np = flat(features_list[2]).cpu().detach().numpy()
    features4_np = flat(features_list[3]).cpu().detach().numpy()
    # 合并四组特征为一个 NumPy 数组 (40000, 64)
    # features = np.vstack([features1_np, features2_np, features3_np, features4_np])
    #
    # # 为每组数据创建标签
    # labels = np.array([0] * 18931 + [1] * 49339 + [2] * 23020 + [
    #     3] * 57016)  # 标签，标识每组数据  18931/148306,49339/148306,23020/148306,57016
    #
    #
    # # 初始化 t-SNE 模型
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    #
    # # 将合并后的特征数据降维至二维
    # features_tsne = tsne.fit_transform(features)  # 输出形状为 (40000, 2)
    # import matplotlib.pyplot as plt
    #
    # # 创建一个颜色映射（不同的标签使用不同的颜色）
    # colors = ['red', 'blue', 'green', 'orange']
    #
    # # 创建图形
    # plt.figure(figsize=(10, 8))
    #
    # # 绘制不同组的数据点
    # for label, color in zip([0, 1, 2, 3], colors):
    #     plt.scatter(features_tsne[labels == label, 0],  # x 坐标
    #                 features_tsne[labels == label, 1],  # y 坐标
    #                 c=color,  # 点的颜色
    #                 label=f'Group {label}',  # 标签
    #                 s=10,  # 点的大小
    #                 alpha=0.6)  # 透明度
    #
    # # 添加标题和图例
    # plt.title('t-SNE Visualization of Four Feature Groups')
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    #######################################
    sample_size = 1000  # 每组采样的样本数量
    features1_sampled = resample(features1_np, n_samples=sample_size, random_state=42)
    features2_sampled = resample(features2_np, n_samples=sample_size, random_state=42)
    features3_sampled = resample(features3_np, n_samples=sample_size, random_state=42)
    features4_sampled = resample(features4_np, n_samples=sample_size, random_state=42)

    # 合并采样后的四组数据
    features_sampled = np.vstack([features1_sampled, features2_sampled, features3_sampled, features4_sampled])

    # 创建标签数组
    labels = np.array([0] * sample_size + [1] * sample_size + [2] * sample_size + [3] * sample_size)

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

    features_tsne = tsne.fit_transform(features_sampled)

    # 可视化降维结果
    colors = ['red', 'blue', 'green', 'orange']
    # plt.figure(figsize=(10, 8))

    # 绘制不同组的数据点
    for label, color in zip([0, 1, 2, 3], colors):
        axes.scatter(features_tsne[labels == label, 0],  # x 坐标
                    features_tsne[labels == label, 1],  # y 坐标
                    c=color,  # 点的颜色
                    label=f'Group {label}',  # 标签
                    s=10,  # 点的大小
                    alpha=0.6)  # 透明度

    # 添加标题、坐标轴标签和图例
    # plt.title('t-SNE Visualization of Four Feature Groups (Sampled)')
    # plt.title(fig_name)
    # plt.xlabel('t-SNE Dimension 1')
    # plt.ylabel('t-SNE Dimension 2')
    # plt.legend()
    # plt.grid(True)
    # # 保存图像（设置dpi调整分辨率）
    # plt.savefig(fig_name+'.png', dpi=300, bbox_inches='tight')
    # plt.show()
