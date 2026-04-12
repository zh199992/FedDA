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
import copy
import torch.nn as nn
import torch.nn.functional as F
print( torch.tensor([5.0,1.0,2.0,3.0]).norm(2))
# Reconstruction Loss
import numpy as np
import matplotlib.pyplot as plt
from utils.data_utils import visualize_features_with_rul, compute_rul_silhouette_score
from models import model
if __name__ == "__main__":
    # m2ymodel = model.GHDR_FL(18).to('cuda')
    input = torch.rand([16,30,18]).to('cuda')
    print(input.shape)
    # output = m2ymodel(input)
    pass
##test git func
##test git+pycharm
    def get_git_revision():
        try:
            # 获取当前 commit 的完整 hash
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd='.',  # 可指定项目根目录
                stderr=subprocess.STDOUT,
                universal_newlines=True
            ).strip()
        except subprocess.CalledProcessError:
            return "unknown"  # 不在 Git 仓库中或 Git 未安装


    # 使用示例
    version = get_git_revision()
    print(f"当前 Git 版本: {version}")

    feature = torch.rand([1024,30,18]).to('cuda')
    label = torch.randint(0,125,[1024,1]).to('cuda')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes.flatten()
    for i in range(4):
        # 展平特征 (b, 30, 18) -> (b, 540)
        features = feature.flatten(start_dim=1)
        rul_labels = label

        # 降维（不绘图）
        embedding = visualize_features_with_rul(features, rul_labels, method='auto')  # 现在这个函数只降维！

        # 在子图上绘制
        ax = copy.deepcopy(axes2[i])
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=rul_labels, cmap='viridis', s=15, alpha=0.7)
        ax.set_title(f"{i+1}", fontsize=10)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.grid(True, linestyle='--', alpha=0.3)#这个ax对象根本没被用到，

        if i == 0:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('RUL')


    plt.tight_layout()
    plt.show()
    # 确保目录存在
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 所有客户端特征图已保存至: {save_path}")
    np.random.seed(42)
    torch.manual_seed(42)

    # 模拟参数
    N = 2000          # 样本数
    D = 50            # 特征维度（类似 C-MAPSS 中间层特征）
    max_rul = 125

    # 生成 RUL 标签：集中在 125（模拟 C-MAPSS）
    # 使用 beta 分布模拟右偏：大部分接近 125
    rul_raw = np.random.beta(0.5, 5, size=N)  # 右偏分布
    rul_labels = max_rul * (1 - rul_raw)      # 映射到 [0, 125]，但集中在 125
    rul_labels = np.clip(rul_labels, 0, max_rul)

    # 生成特征：让特征与 RUL 有一定相关性（模拟“好模型”学到的特征）
    # 简单方式：主成分与 RUL 相关，其余为噪声
    features = np.random.randn(N, D)
    features[:, 0] = rul_labels / max_rul + 0.1 * np.random.randn(N)  # 第一维与 RUL 强相关
    features[:, 1] = (rul_labels / max_rul)**2 + 0.1 * np.random.randn(N)  # 非线性相关

    # 转为 PyTorch 张量（模拟实际训练输出）
    features_tensor = torch.tensor(features, dtype=torch.float32)
    rul_tensor = torch.tensor(rul_labels, dtype=torch.float32)

    print("📊 模拟数据生成完成！")
    print(f"   - 样本数: {N}")
    print(f"   - 特征维度: {D}")
    print(f"   - RUL 范围: [{rul_labels.min():.1f}, {rul_labels.max():.1f}]")
    print(f"   - RUL 均值: {rul_labels.mean():.1f}（应接近 125）")

    # 测试可视化函数
    print("\n🎨 正在测试可视化函数...")
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
def normalized_entropy(probs, n_bins):
    probs = np.array(probs)
    nonzero = probs[probs > 0]
    entropy = -np.sum(nonzero * np.log(nonzero + 1e-12))  # 防止 log(0)
    return entropy / np.log(n_bins)

n_bins = 20
bin_centers = np.arange(1, n_bins + 1)  # 1 to 20

# === 分布 A：目标熵 ≈ 0.868（较均衡）===
# 模拟：大部分 bin 有样本，但尾部（高 RUL）自然衰减
p_A = np.zeros(n_bins)
p_A[:15] = np.exp(-np.linspace(0, 1.8, 15))   # 前15个 bin 有值
p_A[15:] = 0.01  # 尾部保留极小值避免熵过高
p_A = p_A / p_A.sum()

# === 分布 B：目标熵 ≈ 0.766（明显偏斜）===
# 模拟：仅前 10 个 bin 有显著样本，后 10 个几乎为 0
p_B = np.zeros(n_bins)
p_B[:10] = np.exp(-np.linspace(0, 2.5, 10))   # 快速衰减
p_B[10:] = 0.001  # 极小扰动
p_B = p_B / p_B.sum()

H_A = normalized_entropy(p_A, n_bins)
H_B = normalized_entropy(p_B, n_bins)

print(f"分布 A（较均衡）归一化熵: {H_A:.3f}")
print(f"分布 B（偏斜）归一化熵:   {H_B:.3f}")

# 绘图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(bin_centers, p_A, color='steelblue', width=0.8)
plt.title(f'较均衡分布\n归一化熵 = {H_A:.3f}')
plt.xlabel('RUL Bin (1=低RUL, 20=高RUL)')
plt.ylabel('概率')
plt.ylim(0, max(p_A.max(), p_B.max()) * 1.1)

plt.subplot(1, 2, 2)
plt.bar(bin_centers, p_B, color='crimson', width=0.8)
plt.title(f'偏斜分布（尾部缺失）\n归一化熵 = {H_B:.3f}')
plt.xlabel('RUL Bin')
plt.ylabel('概率')
plt.ylim(0, max(p_A.max(), p_B.max()) * 1.1)

plt.tight_layout()
plt.show()

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
