import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import seaborn as sns
import torch
from torch import nn
import torch.nn.functional as F
# import data as data
from torch.utils.data import DataLoader
import torch.nn.init as init
from functools import partial

import torch
import torch.nn as nn


# ----------------------------------------
# 客户端模型: 特征分离自编码器 (FSAE)
# ----------------------------------------
class FSAE(nn.Module):
    """
    Feature Separation Autoencoder (FSAE) for client-side.
    This model consists of two encoders (public and private) and one decoder.
    Only the public encoder's output is shared with the server.
    """

    def __init__(self, input_channels=8):
        super(FSAE, self).__init__()

        # -----------------------------
        # 公共特征编码器 (Public Encoder Eu)
        # 结构: Conv -> ReLU -> Conv -> ReLU
        # 根据Table 2: Convolution 96, 5, 5; Convolution 64, 3, 3
        # 输出维度: [H//15, 64] (例如 XJTU-SY: 2048->136, Milling: 10000->666)
        # -----------------------------
        self.public_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=96, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, stride=3),
            nn.ReLU()
        )

        # -----------------------------
        # 私有特征编码器 (Private Encoder Ev)
        # 结构与公共编码器完全相同
        # -----------------------------
        self.private_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=96, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, stride=3),
            nn.ReLU()
        )

        # -----------------------------
        # 解码器 (Decoder D)
        # 结构: TransposedConv -> ReLU -> TransposedConv -> Sigmoid (或 Tanh)
        # 根据Table 2: Transposed convolution 96, 3, 3; Transposed convolution 8, 5, 5
        # 输入是拼接后的 [Fv, Fu], 所以输入通道数为 64 + 64 = 128
        # -----------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=96, out_channels=8, kernel_size=5, stride=5),
            # 注意: 论文中输出通道为8，但XJTU-SY数据应为4。
            # 在实际训练中，可能需要根据具体数据集调整最后一层的out_channels。
            # 或者在解码器后加一个额外的卷积层来匹配原始输入形状。
            nn.Sigmoid()  # 假设输入数据已归一化到[0,1]
        )

    def forward(self, x):
        """
        Forward pass of the FSAE.

        Args:
            x (torch.Tensor): Input data, shape [batch_size, C, H]

        Returns:
            tuple: (reconstructed_x, Fu, Fv)
                - reconstructed_x: The reconstructed input, shape [batch_size, 8, H_recon]
                - Fu: Public features, shape [batch_size, 64, H_enc]
                - Fv: Private features, shape [batch_size, 64, H_enc]
        """
        # 编码
        Fu = self.public_encoder(x)  # [B, 64, H_enc]
        Fv = self.private_encoder(x)  # [B, 64, H_enc]

        # 拼接公共和私有特征
        # 在通道维度上进行拼接 (dim=1)
        combined_features = torch.cat([Fv, Fu], dim=1)  # [B, 128, H_enc]

        # 解码
        reconstructed_x = self.decoder(combined_features)  # [B, 8, H_recon]

        return reconstructed_x, Fu, Fv
class FSAE_cmapss(nn.Module):
    """
    Feature Separation Autoencoder (FSAE) for client-side.
    This model consists of two encoders (public and private) and one decoder.
    Only the public encoder's output is shared with the server.
    """

    def __init__(self, input_channels=18):
        super(FSAE_cmapss, self).__init__()

        # -----------------------------
        # 公共特征编码器 (Public Encoder Eu)
        # 结构: Conv -> ReLU -> Conv -> ReLU
        # 根据Table 2: Convolution 96, 5, 5; Convolution 64, 3, 3
        # 输出维度: [H//15, 64] (例如 XJTU-SY: 2048->136, Milling: 10000->666)
        # -----------------------------
        self.kernel_size=9
        self.public_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=96, kernel_size=self.kernel_size, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=self.kernel_size, stride=1, padding='same'),
            nn.ReLU()
        )

        # -----------------------------
        # 私有特征编码器 (Private Encoder Ev)
        # 结构与公共编码器完全相同
        # -----------------------------
        self.private_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=96, kernel_size=self.kernel_size, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=96, out_channels=64, kernel_size=self.kernel_size, stride=1, padding='same'),
            nn.ReLU()
        )

        # -----------------------------
        # 解码器 (Decoder D)
        # 结构: TransposedConv -> ReLU -> TransposedConv -> Sigmoid (或 Tanh)
        # 根据Table 2: Transposed convolution 96, 3, 3; Transposed convolution 8, 5, 5
        # 输入是拼接后的 [Fv, Fu], 所以输入通道数为 64 + 64 = 128
        # -----------------------------
        self.decoder = nn.Sequential(
            # nn.Conv1d(in_channels=128, out_channels=96, kernel_size=10, stride=1, padding='same'),
            nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=self.kernel_size, stride=1, padding=4),
            nn.ReLU(),
            # nn.Conv1d(in_channels=96, out_channels=input_channels, kernel_size=10, stride=1, padding='same'),
            nn.ConvTranspose1d(in_channels=96, out_channels=input_channels, kernel_size=self.kernel_size, stride=1,padding=4),
            # 注意: 论文中输出通道为8，但XJTU-SY数据应为4。
            # 在实际训练中，可能需要根据具体数据集调整最后一层的out_channels。
            # 或者在解码器后加一个额外的卷积层来匹配原始输入形状。
        )

    def forward(self, x):
        """
        Forward pass of the FSAE.

        Args:
            x (torch.Tensor): Input data, shape [batch_size, C, H]

        Returns:
            tuple: (reconstructed_x, Fu, Fv)
                - reconstructed_x: The reconstructed input, shape [batch_size, 8, H_recon]
                - Fu: Public features, shape [batch_size, 64, H_enc]
                - Fv: Private features, shape [batch_size, 64, H_enc]
        """
        # 编码
        Fu = self.public_encoder(x)  # [B, 64, H_enc]
        Fv = self.private_encoder(x)  # [B, 64, H_enc]

        # 拼接公共和私有特征
        # 在通道维度上进行拼接 (dim=1)
        combined_features = torch.cat([Fv, Fu], dim=1)  # [B, 128, H_enc]

        # 解码
        reconstructed_x = self.decoder(combined_features)  # [B, 8, H_recon]

        return reconstructed_x, Fu, Fv

    nn.ConvTranspose1d
