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

class ClientCAE(nn.Module):
    def __init__(self):
        super(ClientCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 11, 7, padding=3),  # input: [B, 2, 20000]
            nn.ReLU(),
            nn.Conv1d(11, 6, 7, padding=3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(6, 11, 7, padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(11, 2, 7, padding=3)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class ServerRULPredictor(nn.Module):
    def __init__(self):
        super(ServerRULPredictor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        feat = self.features(x)
        att = self.attention(feat)
        out = self.fc((feat.mean(dim=-1) * att))
        return out.squeeze()
