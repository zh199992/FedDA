import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import torch
from models import model
from models.model_fedrul import ClientCAE
# 示例数据（替换为你的数据）

input=torch.randn(10,2,20000).to('cuda:0')
mymodel=ClientCAE().to('cuda:0')
output=mymodel(input)
print(output[0].shape,output[1].shape)

train_data_dir = os.path.join('/home/zhouheng/project/FedDA/data', "CMAPSSData", 'processed', "18-[0,1]") + '/'
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
