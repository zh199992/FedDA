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

