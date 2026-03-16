import torch.nn.functional as F
import torch
def calculate_l2_diff(model1, model2):
    l2_diff = 0.0
    # 遍历两个模型的所有参数
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        # 确保参数名称和形状一致
        if name1 != name2 or param1.shape != param2.shape:
            raise ValueError("模型参数不匹配！")
        # 计算逐元素平方差并累加
        l2_diff += torch.sum((param1 - param2) ** 2)
    # 开平方得到 L2 范数
    return torch.sqrt(l2_diff).item()


def aggregate_with_sawa(client_models):
    """
    Similarity-Aware Weighted Aggregation (SAWA)

    Args:
        client_models: List of state_dict or parameter tensors (list of OrderedDict or list of torch.Tensor)
                       假设每个元素是 LHDR 的 flatten 参数向量 (e.g., shape=[D])

    Returns:
        aggregated_model: 聚合后的 GHDR 参数向量 (shape=[D])
    """
    def flatten_state_dict(sd):
        return torch.cat([v.flatten() for v in sd.values()])
    flat_params = [flatten_state_dict(lhdr.state_dict()) for lhdr in client_models]
    if len(flat_params) == 1:
        return flat_params[0]

    # Step 1: 将所有客户端参数堆叠成矩阵 [n_clients, D]
    params = torch.stack(flat_params, dim=0)  # shape: (n, D)

    # Step 2: 计算余弦相似度矩阵 [n, n]
    # 先归一化
    norms = torch.norm(params, p=2, dim=1, keepdim=True)  # (n, 1)
    normalized_params = params / (norms + 1e-8)  # 避免除零
    similarity_matrix = torch.mm(normalized_params, normalized_params.t())  # (n, n)

    # Step 3: 计算每个客户端的平均相似度（作为权重）
    avg_similarity = similarity_matrix.mean(dim=1)  # (n,)
    weights = F.softmax(avg_similarity, dim=0)  # 归一化为概率分布，(n,)


    return weights