import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import torch
from models import model, fsae
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reconstruction Loss
def reconstruction_loss(x, x_recon):
    return F.mse_loss(x_recon, x)

# Separation Loss (Inner product minimization)
def separation_loss(Fv, Fu):
    # Fv: [B, C_v, H], Fu: [B, C_u, H]
    inner = torch.einsum('bch,bdh->bcd', Fv, Fu)  # [B, C_v, C_u]
    return torch.mean(torch.abs(inner))

# Similarity Loss using MMD (简化版：用 L2 距离近似)
def similarity_loss(Fu_local, Fu_global):
    return F.mse_loss(Fu_local, Fu_global)

# utils/federated.py
def fed_avg(state_dicts, num_samples):
    """
    state_dicts: list of OrderedDict (model.state_dict())
    num_samples: list of int (number of samples per client)
    """
    total_samples = sum(num_samples)
    avg_state = {}
    for key in state_dicts[0]:
        avg_state[key] = torch.zeros_like(state_dicts[0][key])
        for i, state in enumerate(state_dicts):
            weight = num_samples[i] / total_samples
            avg_state[key] += weight * state[key]
    return avg_state
from models.model_fedrul import ClientCAE
# 示例数据（替换为你的数据）
import torch
import torch.nn as nn
import copy
from models.fsae import FSAE_cmapss
# from utils.federated import fed_avg

def separation_loss(Fv, Fu):
    inner = torch.einsum('bch,bdh->bcd', Fv, Fu)  # [B, C_v, C_u]
    return torch.mean(torch.abs(inner))

def train_fedfsae(clients_data, num_rounds=30, epochs_per_round=1):
    # 初始化全局 public encoder
    global_pub = FSAE_cmapss().public_encoder
    clients = [FSAE_cmapss() for _ in clients_data]

    for round in range(num_rounds):
        client_updates = []
        client_n = []

        for i, (client_model, data_loader) in enumerate(zip(clients, clients_data)):
            optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)
            for _ in range(epochs_per_round):
                for x, _ in data_loader:
                    x = x.transpose(1, 2)
                    x_recon, Fu, Fv = client_model(x)
                    loss_r = nn.MSELoss()(x_recon, x)
                    loss_d = separation_loss(Fv, Fu)

                    # 获取全局 Fu
                    with torch.no_grad():
                        Fu_global = global_pub(x)
                    loss_s = nn.MSELoss()(Fu, Fu_global)  # 简化版

                    loss = loss_r + loss_s + loss_d
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # print(loss.data)
                print(f'epoch:{round}  loss {loss.data}')
            # 上传 public encoder
            client_updates.append(copy.deepcopy(client_model.public_encoder.state_dict()))
            client_n.append(len(data_loader.dataset))

        # Server 聚合
        global_pub.load_state_dict(fed_avg(client_updates, client_n))

        # 下发全局模型（用于下一轮 similarity loss）
        for client in clients:
            client.global_pub = copy.deepcopy(global_pub)

    return clients, global_pub

input=torch.randn(50,18, 30).to('cuda:0')
mymodel=fsae.FSAE_cmapss().to('cuda:0')
# mymodel=model.GHDR_FL(18).to('cuda:0')
# input=torch.randn(10,2,20000).to('cuda:0')
# mymodel=ClientCAE().to('cuda:0')
output=mymodel(input)
print(output[0].shape,output[1].shape)



train_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', "CMAPSSData", 'processed', "18-[0,1]") + '/'
batch_size=1024
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_client_data
import argparse

parser = argparse.ArgumentParser()
# general  添加参数
parser.add_argument("-random_seed", "--random_seed", type=int, default=42)
parser.add_argument('-d', "--directory", type=str, default="0821")
parser.add_argument('-aim', "--aim", type=str, default="debug")  # 训练目的
parser.add_argument('-data', "--dataset", type=str, default="CMAPSSData")
parser.add_argument('-m', "--model_name", type=str, default="cnn1D", choices=["cnn1D", "lstm", "FedRUL"])
# parser.add_argument('-cloudm', "--model_name", type=str, default="cnn1D",choices=["cnn1D","lstm","FedRUL"])
parser.add_argument('-dp', "--dp", type=str, default="18-[0,1]",
                    choices=["14-[-1,1]", "18-[0,1]", "14-[0,1]", "18-[-1,1]"], help="dataprocessing")
parser.add_argument('-algo', "--algorithm", type=str, default="FedDA",  ##这个参数不传入server
                    choices=["centralized", "local", "localiid", "FedAvg", "FedAvgiid", "GHDR", "FedDA", "ablation1",
                             "ablation2", "finetune", "dann"])
parser.add_argument('-o_c', "--optimizer_client", type=str, default="adam", choices=["adam", "adamod", "sgd"])
parser.add_argument('-o_s', "--optimizer_server", type=str, default="adam", choices=["adam", "adamod", "sgd"])
parser.add_argument('-bs_c', "--batch_size_client", type=int, default=1024)
parser.add_argument('-bs_s', "--batch_size_server", type=int, default=1024)
parser.add_argument('-nc', "--num_clients", type=int, default=4,
                    help="Total number of clients")
parser.add_argument('-gr_i', "--global_rounds_init", type=int, default=0)
parser.add_argument('-gr', "--global_rounds", type=int, default=10)
parser.add_argument('-early_stop', "--early_stop", type=bool, default=True)
# parser.add_argument('-le', "--local_epochs", type=str, default='50,5',
#                     help="Multiple update steps in one local epoch.")
parser.add_argument('-le', "--local_epochs", type=int, default=0,
                    help="Multiple update steps in one local epoch.")
parser.add_argument('-se', "--server_epochs", type=int, default=10)
# parser.add_argument('-clr', "--local_learning_rate", type=str, default='0.001,0.001')
# parser.add_argument('-slr', "--server_learning_rate", type=str, default='0.001,0.001')
parser.add_argument('-clr', "--local_learning_rate", type=float, default=0.001)
parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.001)
# parser.add_argument('-did', "--device_id", type=str, default="0")#?
parser.add_argument('-sches', "--server_schedule", type=bool, default=False)
parser.add_argument('-schec', "--client_schedule", type=bool, default=False)
parser.add_argument('-clips', "--server_clip", type=bool, default=False)
parser.add_argument('-clipc', "--client_clip", type=bool, default=False)
parser.add_argument('-ws', "--window_size", type=int, default=30)  # 删了train window还是test window?
parser.add_argument('-lrd_c', "--client_lr_decay", type=bool, default=False)
parser.add_argument('-lrd_s', "--server_lr_decay", type=bool, default=False)

parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

parser.add_argument('-conv_init', "--conv_init", type=str, default="kaiming_uniform",
                    choices=["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "normal",
                             "uniform"])
parser.add_argument('-gru_init', "--gru_init", type=str, default="kaiming_uniform",
                    choices=["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "normal",
                             "uniform"])
parser.add_argument('-linear_init', "--linear_init", type=str, default="kaiming_uniform",
                    choices=["kaiming_uniform", "kaiming_normal", "xavier_uniform", "xavier_normal", "normal",
                             "uniform"])
parser.add_argument('-F_FedAvg', "--F_FedAvg", type=bool, default=False)
parser.add_argument('-EDI_FedAvg', "--EDI_FedAvg", type=bool, default=False)  # 不freeze也不fedavg就是个性化
parser.add_argument('-P_FedAvg', "--P_FedAvg", type=bool, default=False)
parser.add_argument('-EDI_Freeze', "--EDI_Freeze", type=bool, default=False)
parser.add_argument('-EDS', "--EDS", type=bool, default=False)  # 影响模型的forward
parser.add_argument('-fedeval', "--fedeval", type=bool, default=False)
parser.add_argument('-DA_loss', type=str, default="adv+mmd", choices=["adv+mmd", "adv", "mmd", "none"])
parser.add_argument('-lambda_mmd', "--lambda_mmd", type=float, default=0.05)
parser.add_argument('-gamma', "--gamma", type=float, default=0.05)

args = parser.parse_args()
train_data1 = read_client_data('CMAPSSData', 1, args, is_train=True)
train_data2 = read_client_data('CMAPSSData', 2, args, is_train=True)
clients_data=[
    DataLoader(train_data1, batch_size, drop_last=False, shuffle=True),
    DataLoader(train_data2, batch_size, drop_last=False, shuffle=True)
]
train_fedfsae(clients_data)

X = torch.load(train_data_dir + "train_FD002" + "feature" + str(18) + '.pt')
indices = torch.randperm(X.size(0))[:1000]
# 使用索引采样
sampled_data = X[indices].to('cuda:0')
y=torch.load(train_data_dir + "train_FD002" +"label"+ str(18)+'.pt')[indices].numpy()



best_model_path = '/home/zhouheng/project/FedDA/models/weights/' + 'local' + '/' + 'GHDR_FL18' + f"_{2}"
global_model0 = model.GHDR_FL(18).to('cuda:0')
global_model0.load_state_dict(torch.load(best_model_path))
global_model=model.GHDR_test(18).to('cuda:0')
global_model.mode = global_model0.mode
global_model.input_size = global_model0.input_size
global_model.filter_num = global_model0.filter_num
global_model.filter_length = global_model0.filter_length
global_model.F=global_model0.F
global_model.LHDR=global_model0.LHDR
global_model.GRU=global_model0.unique[0]
_,shallow,middle=global_model(sampled_data)
# t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
sampled_data=sampled_data.reshape(sampled_data.size(0),-1)
shallow=shallow.reshape(shallow.size(0),-1)
middle=middle.reshape(middle.size(0),-1)
raw_embedded = tsne.fit_transform(sampled_data.cpu().detach().numpy())
shallow_embedded = tsne.fit_transform(shallow.cpu().detach().numpy())
middle_embedded = tsne.fit_transform(middle.cpu().detach().numpy())

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
norm = mcolors.Normalize(vmin=0, vmax=125)
cmap = cm.get_cmap('viridis')
# 第一组图
sc1 = axes[0].scatter(raw_embedded[:, 0], raw_embedded[:, 1], c=y, cmap=cmap, norm=norm, s=10)
axes[0].set_title("raw_data")
axes[0].set_xlabel("Dim 1")
axes[0].set_ylabel("Dim 2")

sc2 = axes[1].scatter(shallow_embedded[:, 0], shallow_embedded[:, 1], c=y, cmap=cmap, norm=norm, s=10)
axes[1].set_title("shallow_Feature")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")

# 第二组图
sc3 = axes[2].scatter(middle_embedded[:, 0], middle_embedded[:, 1], c=y, cmap=cmap, norm=norm, s=10)
axes[2].set_title("middle_feature")
axes[2].set_xlabel("Dim 1")
axes[2].set_ylabel("Dim 2")

# 添加公共colorbar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='vertical', fraction=0.02)
cbar.set_label("Label (0–125)")

plt.tight_layout()
plt.show()
# # 创建归一化器和 colormap
# norm = mcolors.Normalize(vmin=0, vmax=125)
# cmap = cm.get_cmap('viridis')
#
# # 显式创建图形和坐标轴
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # 绘制散点图
# sc = ax.scatter(shallow_embedded[:, 0], shallow_embedded[:, 1], c=y, cmap=cmap, norm=norm, s=10)
#
# # 添加 colorbar，并与 ax 绑定
# cbar = fig.colorbar(sc, ax=ax)
# cbar.set_label('Label (0–125)')
#
# ax.set_title("t-SNE Visualization with Labels 0–125")
# ax.set_xlabel("Dimension 1")
# ax.set_ylabel("Dimension 2")
#
# plt.tight_layout()
# plt.show()
