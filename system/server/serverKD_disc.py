import torch
import torch.nn as nn
import torch.optim as optim
# from models import RULPredictor
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np


#===============================================================================
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# 模拟四个子数据集（FD001~FD004）
def load_cmswss_client_data(client_id, num_samples=500, input_dim=14):
    np.random.seed(client_id)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    # 模拟不同 domain 的 RUL 分布（例如 FD004 更长寿命）
    base_rul = np.random.exponential(scale=80 if client_id in [2, 3] else 50, size=num_samples)#指数分布中采样
    y = np.clip(base_rul, 0, 125).astype(np.float32)
    return X, y


def get_client_dataloader(client_id, batch_size=32, num_bins=20):
    X, y_cont = load_cmswss_client_data(client_id)
    y_disc = discretize_rul(y_cont, num_bins=num_bins)

    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y_disc, dtype=torch.long),
        torch.tensor(y_cont)  # 保留连续值用于评估
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
#===============================================================================
def discretize_rul(rul, num_bins=20, max_rul=125):#输入(N,)
    """
    将连续 RUL 离散化为分类标签（用于蒸馏）
    """
    rul = np.clip(rul, 0, max_rul)#(N,)
    bins = np.linspace(0, max_rul, num_bins + 1)#生成bin边界 num_bins个bin
    labels = np.digitize(rul, bins) - 1#(N,)个从0开始的桶索引
    labels = np.clip(labels, 0, num_bins - 1)#确保<=19
    return labels

def compute_soft_logits(logits, temperature=2.0):
    """
    计算软化后的概率分布（用于蒸馏）
    """
    return torch.softmax(logits / temperature, dim=-1)

def rul_from_logits(logits, num_bins=20, max_rul=125):
    """
    从 logits 恢复 RUL 预测值（用于评估）
    """
    probs = torch.softmax(logits, dim=-1)
    bin_centers = torch.linspace(0, max_rul, num_bins).to(logits.device)
    pred_rul = (probs * bin_centers).sum(dim=-1)
    return pred_rul


#===============================================================================
# ========== 超参数 ==========
NUM_CLIENTS = 4
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
NUM_BINS = 20
TEMPERATURE = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 初始化 clients ==========
clients = []
optimizers = []
for i in range(NUM_CLIENTS):
    model = RULPredictor(num_bins=NUM_BINS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    clients.append(model)
    optimizers.append(optimizer)

# ========== 联邦训练 ==========
for round_idx in range(NUM_ROUNDS):
    print(f"\n=== Round {round_idx + 1}/{NUM_ROUNDS} ===")

    # Step 1: 每个 client 本地训练
    all_soft_logits = []  # 存储每个 client 的 soft logits
    client_eval_ruls = []
    client_true_ruls = []

    for cid in range(NUM_CLIENTS):
        model = clients[cid]
        optimizer = optimizers[cid]
        dataloader = get_client_dataloader(cid, batch_size=BATCH_SIZE, num_bins=NUM_BINS)

        model.train()
        for epoch in range(LOCAL_EPOCHS):
            for x, y_disc, y_cont in dataloader:
                x, y_disc = x.to(DEVICE), y_disc.to(DEVICE)
                logits, _ = model(x)
                loss = nn.CrossEntropyLoss()(logits, y_disc)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Step 2: 生成 soft logits 用于上传
        model.eval()
        soft_logit_list = []
        true_rul_list = []
        pred_rul_list = []
        with torch.no_grad():
            for x, _, y_cont in dataloader:
                x, y_cont = x.to(DEVICE), y_cont.to(DEVICE)
                logits, _ = model(x)
                soft_logits = compute_soft_logits(logits, temperature=TEMPERATURE)
                soft_logit_list.append(soft_logits.cpu())
                true_rul_list.append(y_cont.cpu())
                pred_rul = rul_from_logits(logits, num_bins=NUM_BINS)
                pred_rul_list.append(pred_rul.cpu())

        # 合并所有 batch
        soft_logits_cat = torch.cat(soft_logit_list, dim=0)  # [N, num_bins]
        all_soft_logits.append(soft_logits_cat)
        client_eval_ruls.append(torch.cat(pred_rul_list))
        client_true_ruls.append(torch.cat(true_rul_list))

    # Step 3: Server 聚合 soft logits（按样本对齐）#得保证client有公共样本=========================
    # 注意：实际中需对齐样本 ID；此处假设所有 client 有相同数量/顺序样本（简化）
    # 更稳健做法：只聚合 validation set 上的 logits
    global_soft_logits = torch.stack(all_soft_logits, dim=0).mean(dim=0)  # [N, num_bins]

    # Step 4: 下发 global_soft_logits，蒸馏训练
    for cid in range(NUM_CLIENTS):
        model = clients[cid]
        optimizer = optimizers[cid]
        dataloader = get_client_dataloader(cid, batch_size=BATCH_SIZE, num_bins=NUM_BINS)

        model.train()
        for x, _, _ in dataloader:
            x = x.to(DEVICE)
            logits, _ = model(x)
            soft_pred = compute_soft_logits(logits, temperature=TEMPERATURE)

            # 获取对应样本的 global soft target（简化：按顺序）
            # 实际应用中应使用固定 validation set 并缓存索引
            idx = ...  # 此处省略索引对齐逻辑（见下方说明）
            global_target = global_soft_logits[idx].to(DEVICE)

            # 蒸馏损失（KL 散度）
            loss_kd = nn.KLDivLoss(reduction='batchmean')(
                torch.log(soft_pred + 1e-8),
                global_target
            )
            optimizer.zero_grad()
            loss_kd.backward()
            optimizer.step()

    # Step 5: 评估（可选）
    if (round_idx + 1) % 10 == 0:
        for cid in range(NUM_CLIENTS):
            pred = client_eval_ruls[cid].numpy()
            true = client_true_ruls[cid].numpy()
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            print(f"Client {cid} RMSE: {rmse:.2f}")