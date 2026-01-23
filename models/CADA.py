# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FedCADA(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=32, num_layers=5,seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.F = nn.LSTM(input_dim, hidden_dim, num_layers-3,batch_first=True)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
        #                     batch_first=True, bidirectional=True)
        self.E_DI = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )
        self.InfoNCEHead = nn.ModuleList([
            nn.Linear(hidden_dim, input_dim) for _ in range(seq_len)
        ])#输入(n,30,14) 输出(30,n,30,14),如果第k个head，得到的正样本是(k,n,k,14),负样本是(k,n,[:k],14)+(k,n,[k+1:],14)

    def forward(self, x):
        shallow, _ = self.F(x)
        f_T, _ = self.E_DI(shallow)
        y_pred = self.predictor(f_T[:, -1, :]) # [B, hidden*2]

        B, K, M = x.shape
        assert K == self.seq_len and M == self.input_dim
        total_loss = 0.0
        for k in range(self.seq_len):
            q_k = self.theta_layers[k](f_T)      # [B, input_dim]
            x_k = x[:, k, :]                   # [B, input_dim]

            # 构造相似度：q_k 与 同样本所有 x_j 的点积（j=0,...,K-1）
            # 正样本：j == k
            # 负样本：j != k
            # 计算 q_k 与 X_T 中所有时间步的点积
            # sim[i, j] = q_k[i] · x_j[i]  （注意：只与自己样本的 x_j 比较）
            # 所以不能用 mm(q_k, X_T[:, j].T)，因为那是跨样本

            # 正确做法：逐样本计算（或使用广播）
            # q_k: [B, M], X_T: [B, K, M] → 点积在 M 上，结果 [B, K]
            sim = torch.einsum('bm,bkm->bk', q_k, x)  # [B, K] 每个样本在第k步的transformed feature和原始序列每一步的sim
            # sim = sim / 0.1  # temperature

            # 标签：每个样本的正样本在位置 k
            labels = torch.full((B,), k, device=f_T.device, dtype=torch.long)

            loss_k = F.cross_entropy(sim, labels)
            total_loss += loss_k

        return y_pred, f_T, total_loss / K



class InfoNCEHead(nn.Module):
    def __init__(self, feat_dim=32, input_dim=18, seq_len=30):
        super().__init__()
        # 每个时间步 k 有独立 Θ_k: R^d -> R^M
        self.theta_layers = nn.ModuleList([
            nn.Linear(feat_dim, input_dim) for _ in range(seq_len)
        ])
        self.seq_len = seq_len
        self.input_dim = input_dim

    def forward(self, f_T, X_T):
        """
        f_T: [B, d]
        X_T: [B, K, M]
        Returns: scalar InfoNCE loss (averaged over k)
        """
        B, K, M = X_T.shape
        assert K == self.seq_len and M == self.input_dim

        total_loss = 0.0
        for k in range(K):
            q_k = self.theta_layers[k](f_T)      # [B, M]
            x_k = X_T[:, k, :]                   # [B, M]

            # 构造相似度矩阵：q_k 与 batch 内所有 x_k 的点积
            # 正样本：(q_k[i], x_k[i])
            # 负样本：(q_k[i], x_k[j≠i])
            sim = torch.mm(q_k, X_T[:, k, :].T)  # [B, B]
            sim = sim / 0.1  # temperature

            labels = torch.arange(B, device=f_T.device)
            loss_k = F.cross_entropy(sim, labels)
            total_loss += loss_k

        return total_loss / K