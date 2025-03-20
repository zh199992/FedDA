import torch.nn as nn
import torch
# m = nn.Conv1d(16, 33, 3, stride=2)
# input = torch.randn(20, 16, 50)
# print(m(input).size())
# a=torch.tensor([[[[1,2,3],[4,5,6]]]],dtype=torch.float32)
# conv=nn.Conv1d(in_channels=1,out_channels=2,kernel_size=2)
# pool=nn.AdaptiveMaxPool1d(output_size=1)
# print(pool(a))
# print(conv(a))
import tensorflow as tf
import numpy as np
inputt = data = tf.constant(
    [[[1.0] * 14 for _ in range(30)]],  # 30 行 14 列的全零矩阵
    dtype=tf.float32
)
print(inputt.shape)
conv_layer = tf.keras.layers.Conv1D(filters=10, kernel_size=3, activation='tanh',                            kernel_initializer='glorot_uniform')
x = conv_layer(inputt)
print(x.shape)      # 输出: (1,30,10)

kernel, bias = conv_layer.get_weights()
conv_layer2 = tf.keras.layers.Conv1D(filters=10, kernel_size=10, activation='tanh', padding='same',
                           kernel_initializer='glorot_uniform')
x = conv_layer2(x)
print(x.shape)      # 输出: (10,)
kernel2, bias2 = conv_layer.get_weights()
print("卷积核形状:", kernel.shape)  # 输出: (10, 14, 10) 10步，14个通道 10个卷积核
print("偏置形状:", bias.shape)      # 输出: (10,)
print("卷积核形状:", kernel2.shape)  # 输出: (10, 14, 10)
print("偏置形状:", bias2.shape) #(10,)

aa=torch.randn(10,1,30,14) #batch channel time feature
conv=nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(10,1),padding='same')
print(conv(aa).shape)
conv2=nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(10,1))
print(conv2(aa).shape)

aa=torch.randn(18,14,30)   # batch channel time
conv=nn.Conv1d(in_channels=14,out_channels=10,kernel_size=3)  #(10,10,28)
print(conv(aa).shape)

fl=nn.Flatten()
print(fl(aa).shape)


def __getitem__(idx):
    for i, size in enumerate([10,15,20,25]):
        if idx < size:
            return i  # 返回数据、标签和域索引
        idx -= size

print(__getitem__(15))
from itertools import zip_longest
for idx, i in enumerate(range(10)):
    print(idx,i)

print(aa.size(0))
bbbbb=torch.tensor([[0,1],[1,1],[2,1],[3,1],[0,5]])
bbb=torch.unique(bbbbb)
print(bbbbb.size())
ccc=torch.tensor([[0],[1],[2],[3],[0]])
print(ccc[ccc==2])
