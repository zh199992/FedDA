import torch
import torch.nn.functional as F


def custom_loss(h, y_hat, tau=0.5, L=1.0):
    """
    h: (N, D) - embeddings from the network
    y_hat: (N,) - predicted or target labels (same type as used in loss)
    tau: temperature for embedding similarity
    L: temperature for label distance

    Returns:
        scalar loss
    """
    N = h.size(0)

    # Step 1: 计算所有样本对之间的欧氏距离和标签距离
    # |h_i - h_j|^2
    h_dist = torch.cdist(h, h, p=2)  # (N, N)

    # |y_i - y_j|
    y_hat_diff = torch.abs(y_hat.unsqueeze(1) - y_hat.unsqueeze(0))  # (N, N)

    # Step 2: 分别计算分子和分母项
    # mask to remove i == j
    mask = ~torch.eye(N, dtype=torch.bool, device=h.device)

    # Numerator
    num = torch.exp(-tau * h_dist) * torch.exp(-y_hat_diff / L)
    num = num.masked_select(mask).reshape(N, N - 1).sum(dim=1)

    # Denominator
    denom = torch.exp(-y_hat_diff / L)
    denom = denom.masked_select(mask).reshape(N, N - 1).sum(dim=1)

    # Final loss
    loss = -torch.log(num / denom + 1e-8).mean()
    return loss
