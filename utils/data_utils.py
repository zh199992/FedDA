import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.utils import resample
from utils.root import find_project_root
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap  # 可选，需 pip install umap-learn
from sklearn.preprocessing import KBinsDiscretizer
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
# def read_client_data(dataset, idx, args,  is_train=True):  #调用read_data   输出元组列表
#     current_directory = os.getcwd()
#     root_dir = find_project_root('FedDA')
#     if is_train:
#         train_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'
#
#         train_feature = torch.load(train_data_dir + "train_FD00" +str(idx)+"feature"+ str(args.window_size)+'.pt')
#         train_label = torch.load(train_data_dir + "train_FD00" +str(idx)+"label"+ str(args.window_size)+'.pt')
#         train_set = MyDataset(train_feature, train_label)
#
#         return train_set
#
#     else:
#         test_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'
#         test_feature = torch.load(test_data_dir + "RUL_FD00" +str(idx)+str(args.window_size)+'.pt')
#         test_label = torch.load(test_data_dir + "test_FD00" +str(idx)+ str(args.window_size)+'.pt')
#         test_set = MyDataset(test_feature, test_label)
#
#         return test_set

def read_client_data(dataset, idx, args,  is_train=True, train_ratio=1.0):  #调用read_data   输出元组列表
    current_directory = os.getcwd()
    root_dir = find_project_root('FedDA')
    if is_train:
        train_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'

        train_feature = torch.load(train_data_dir + "train_FD00" +str(idx)+"feature"+ str(args.window_size)+'.pt')
        train_label = torch.load(train_data_dir + "train_FD00" +str(idx)+"label"+ str(args.window_size)+'.pt')
        full_train_set = MyDataset(train_feature, train_label)

        if train_ratio < 1.0 and train_ratio > 0.0:
            total_size = len(full_train_set)
            subset_size = int(total_size * train_ratio)
            remainder = total_size - subset_size
            # 随机划分
            subset, _ = random_split(full_train_set, [subset_size, remainder])
            return subset
        else:
            return full_train_set

    else:
        test_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'
        test_feature = torch.load(test_data_dir + "RUL_FD00" +str(idx)+str(args.window_size)+'.pt')
        test_label = torch.load(test_data_dir + "test_FD00" +str(idx)+ str(args.window_size)+'.pt')
        test_set = MyDataset(test_feature, test_label)

        return test_set

def read_client_data_iid(dataset, split_sizes, start, idx, args,  is_train=True):  #调用read_data   输出元组列表
    current_directory = os.getcwd()
    root_dir = find_project_root('FedDA')
    if is_train:
        train_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'

        # train_feature = torch.load(train_data_dir + "train_FD00" +str(idx)+"feature"+ str(args.window_size)+'.pt')
        # train_label = torch.load(train_data_dir + "train_FD00" +str(idx)+"label"+ str(args.window_size)+'.pt')
        train_feature = torch.load(train_data_dir + "train_FD001" +"feature"+ str(args.window_size)+'.pt')[start:start+split_sizes[idx-1]]
        train_label = torch.load(train_data_dir + "train_FD001" +"label"+ str(args.window_size)+'.pt')[start:start+split_sizes[idx-1]]
        train_set = MyDataset(train_feature, train_label)

        return train_set

    else:#应该整个还是四分之一？ 两个都用吧
        test_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'
        # test_feature = torch.load(test_data_dir + "RUL_FD00" +str(idx)+str(args.window_size)+'.pt')
        # test_label = torch.load(test_data_dir + "test_FD00" +str(idx)+ str(args.window_size)+'.pt')
        test_feature = torch.load(test_data_dir + "RUL_FD001" + str(args.window_size) + '.pt')[start:start+split_sizes[idx-1]]
        test_label = torch.load(test_data_dir + "test_FD001" + str(args.window_size) + '.pt')[start:start+split_sizes[idx-1]]
        test_set = MyDataset(test_feature, test_label)

        return test_set

def read_client_data_centralized(dataset, args,  is_train=True):  #调用read_data   输出元组列表
    current_directory = os.getcwd()
    root_dir = find_project_root('FedDA')
    if is_train:
        train_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'
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
        test_data_dir = os.path.join(root_dir, 'data', dataset, 'processed',args.dp)+'/'
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
    root_dir = find_project_root('FedDA')
    data_dir = os.path.join(root_dir, 'data', dataset, 'processed', args.dp) + '/'
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
def str_to_bool(value):
    true_values = {'true', 'yes', '1', 't'}
    false_values = {'false', 'no', '0', 'f'}
    if value.lower() in true_values:
        return True
    elif value.lower() in false_values:
        return False


def visualize_features_with_rul(features, rul_labels, method='auto', n_components=2):
    """仅降维，不绘图"""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(rul_labels, torch.Tensor):
        rul_labels = rul_labels.detach().cpu().numpy()

    features = features.astype(np.float32)
    rul_labels = rul_labels.astype(np.float32)

    if method == 'auto':
        pca_temp = PCA(n_components=min(10, features.shape[1]))
        pca_temp.fit(features)
        cumsum_ratio = np.cumsum(pca_temp.explained_variance_ratio_)
        if cumsum_ratio[-1] > 0.85:
            method = 'pca'
        else:
            method = 'umap' if 'umap' in globals() else 'tsne'

    if method == 'pca':
        reducer = PCA(n_components=n_components)
        embedding = reducer.fit_transform(features)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=42, n_iter=300)
        embedding = reducer.fit_transform(features)  # 注意：TSNE 很慢，可考虑 subsample
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(features)
    else:
        raise ValueError("method must be 'pca', 'tsne', 'umap', or 'auto'")

    return embedding  # ← 只返回 embedding，不绘图！#ndarray


def compute_rul_silhouette_score(features, rul_labels, n_bins=10):
    """
    计算基于 RUL 分箱的 Silhouette Score，用于衡量特征空间中 RUL 相似样本的聚集程度。

    Args:
        features: (N, D) array
        rul_labels: (N,) array
        n_bins: 将 RUL 分为多少个 bin（伪类别）

    Returns:
        float: Silhouette Score ∈ [-1, 1]，越高越好
    """
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(rul_labels, torch.Tensor):
        rul_labels = rul_labels.detach().cpu().numpy()

    # 离散化 RUL 为伪类别
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    pseudo_labels = discretizer.fit_transform(rul_labels.reshape(-1, 1)).flatten().astype(int)

    # 至少需要2个类别
    if len(np.unique(pseudo_labels)) < 2:
        return -1.0

    try:
        score = silhouette_score(features, pseudo_labels, metric='euclidean')
        return float(score)
    except:
        return -1.0


def plot_all_clients_features(uploaded_middle_features, uploaded_labels, titles=None, save_path="all_clients_features.png", figsize=(12, 10)):
    n_clients = len(uploaded_middle_features)
    assert n_clients <= 4, "最多支持4个客户端"
    if titles is None:
        titles = [f"Client {i + 1}" for i in range(n_clients)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    all_scores = []

    for i in range(n_clients):
        # 展平特征 (b, 30, 18) -> (b, 540)
        features = uploaded_middle_features[i].flatten(start_dim=1)
        rul_labels = uploaded_labels[i].detach().cpu().numpy()

        # 计算聚类分数
        score = compute_rul_silhouette_score(features, rul_labels, n_bins=10)
        all_scores.append(score)
        print(f"✅ Client {i+1} RUL-based Silhouette Score: {score:.4f}")

        # 降维（不绘图）
        embedding = visualize_features_with_rul(features, rul_labels, method='auto')  # 现在这个函数只降维！

        # 在子图上绘制
        ax = axes[i]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=rul_labels, cmap='viridis', s=15, alpha=0.7)
        ax.set_title(f"{titles[i]} | RSS={score:.3f}", fontsize=10)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, linestyle='--', alpha=0.3)

        if i == 0:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('RUL')

    # 隐藏多余子图
    for j in range(n_clients, 4):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 所有客户端特征图已保存至: {save_path}")#阻塞了 没到这一步
    return all_scores

def regresssion_feature(features_tensor, rul_tensor):
    print("\n🎨 正在运行可视化函数...")
    embedding = visualize_features_with_rul(
        features_tensor,
        rul_tensor,
        method='auto',
        title="Simulated C-MAPSS Features"
    )

    # 测试聚类指标
    print("\n📈 正在计算 RUL 聚类质量指标...")
    score = compute_rul_silhouette_score(features_tensor, rul_tensor, n_bins=10)
    print(f"✅ RUL-based Silhouette Score: {score:.4f}")
    print("   - 越接近 1 表示 RUL 相似样本越聚集")
    print("   - 负值表示无聚类结构")

    print("\n🎉 所有函数运行成功！")