from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os


train_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', "CMAPSSData", 'processed', "18-[0,1]") + '/'
train_feature = torch.load(train_data_dir + "train_FD001" + "feature" + str(18) + '.pt')
# total = train_feature.size(0)
# base = total // 4
# remainder = total % 4
# split_sizes = [base + 1 if i < remainder else base for i in range(4)]
# parts = {}
# start=0
# for i, size in enumerate(split_sizes):
#     parts[f'part_{i+1}'] = train_feature[start:start + size]  # 动态生成键名
#     start += size
#
# parts['part_1']=parts['part_1'].reshape(parts['part_1'].size(0), -1)
# parts['part_2']=parts['part_2'].reshape(parts['part_2'].size(0), -1)
# parts['part_3']=parts['part_3'].reshape(parts['part_3'].size(0), -1)
# parts['part_4']=parts['part_4'].reshape(parts['part_4'].size(0), -1)
n = train_feature.size(0)
base, remainder = divmod(n, 4)
split_sizes = [base + 1] * remainder + [base] * (4 - remainder)

# 随机打乱分割顺序
indices = torch.randperm(4)
split_sizes = [split_sizes[i] for i in indices]  # 例如：n=10 → [3,3,2,2] 的随机排列

# 生成随机索引并分割
perm = torch.randperm(n)
parts = {}
start = 0
for i,size in enumerate(split_sizes):
    idx = perm[start: start + size]
    part = train_feature[idx]  # 提取子张量，形状为 (size, 30, 18)
    parts[f'part_{i + 1}'] = part
    start += size
parts['part_1']=parts['part_1'].reshape(parts['part_1'].size(0), -1)
parts['part_2']=parts['part_2'].reshape(parts['part_2'].size(0), -1)
parts['part_3']=parts['part_3'].reshape(parts['part_3'].size(0), -1)
parts['part_4']=parts['part_4'].reshape(parts['part_4'].size(0), -1)

train_feature1 = torch.load(train_data_dir + "train_FD00" + str(1) + "feature" + str(18) + '.pt')
train_feature2 = torch.load(train_data_dir + "train_FD00" + str(2) + "feature" + str(18) + '.pt')
train_feature3 = torch.load(train_data_dir + "train_FD00" + str(3) + "feature" + str(18) + '.pt')
train_feature4 = torch.load(train_data_dir + "train_FD00" + str(4) + "feature" + str(18) + '.pt')

data_a_flat = train_feature1.reshape(train_feature1.size(0), -1)
data_b_flat = train_feature2.reshape(train_feature2.size(0), -1)
data_c_flat = train_feature3.reshape(train_feature3.size(0), -1)
data_d_flat = train_feature4.reshape(train_feature4.size(0), -1)
# train_label = torch.load(train_data_dir + "train_FD00" + str(1) + "label" + str(18) + '.pt')

test_feature1 = torch.load(train_data_dir + "RUL_FD00" + str(1) +  str(18) + '.pt')
test_feature2 = torch.load(train_data_dir + "RUL_FD00" + str(2) +  str(18) + '.pt')
test_feature3 = torch.load(train_data_dir + "RUL_FD00" + str(3) +  str(18) + '.pt')
test_feature4 = torch.load(train_data_dir + "RUL_FD00" + str(4) +  str(18) + '.pt')

test_a_flat = test_feature1.reshape(test_feature1.size(0), -1)
test_b_flat = test_feature2.reshape(test_feature2.size(0), -1)
test_c_flat = test_feature3.reshape(test_feature3.size(0), -1)
test_d_flat = test_feature4.reshape(test_feature4.size(0), -1)

pca = PCA(n_components=2)
# proj_a = pca.fit_transform(data_a_flat)
# proj_b = pca.transform(data_b_flat)
# proj_c = pca.transform(data_c_flat)
# proj_d = pca.transform(data_d_flat)
# proj_e = pca.transform(test_a_flat)
# proj_f = pca.transform(test_b_flat)
# proj_g = pca.transform(test_c_flat)
# proj_h = pca.transform(test_d_flat)
proj_1 = pca.fit_transform(parts['part_1'])
proj_2 = pca.transform(parts['part_2'])
proj_3 = pca.transform(parts['part_3'])
proj_4 = pca.transform(parts['part_4'])

plt.figure(figsize=(10, 6))
# plt.scatter(proj_a[:, 0], proj_a[:, 1], label='Dataset A', alpha=0.5, s=10)
# plt.scatter(proj_b[:, 0], proj_b[:, 1], label='Dataset B', alpha=0.5, s=10)
# plt.scatter(proj_c[:, 0], proj_c[:, 1], label='Dataset C', alpha=0.5, s=10)
# plt.scatter(proj_d[:, 0], proj_d[:, 1], label='Dataset D', alpha=0.5, s=10)
# plt.scatter(proj_e[:, 0], proj_e[:, 1], label='test A', alpha=0.5, s=10)
# plt.scatter(proj_f[:, 0], proj_f[:, 1], label='test B', alpha=0.5, s=10)
# plt.scatter(proj_g[:, 0], proj_g[:, 1], label='test C', alpha=0.5, s=10)
# plt.scatter(proj_h[:, 0], proj_h[:, 1], label='test D', alpha=0.5, s=10)
plt.scatter(proj_1[:, 0], proj_1[:, 1], label='trainA_1', alpha=0.5, s=10)
plt.scatter(proj_2[:, 0], proj_2[:, 1], label='trainA_1', alpha=0.5, s=10)
plt.scatter(proj_3[:, 0], proj_3[:, 1], label='trainA_1', alpha=0.5, s=10)
plt.scatter(proj_4[:, 0], proj_4[:, 1], label='trainA_1', alpha=0.5, s=10)



plt.title('train_set001 randomly splited into 4')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()