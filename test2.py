import torch.nn as nn
import torch
import numpy
import time

import models.model
from distutils.util import strtobool
print(int(1.1),int(1.9),int(-1.1),int(-1.9))
a = "true"
if type(a) == str:
    print("字符串")
    a = bool(strtobool(a))
device = torch.device('cuda:1')
print(float('1,1'.split(',')[0]))
dummy = torch.empty(1, device=device)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

mymodel = models.model.conv_DANN2(18).to(device)
print(f"模型后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

optimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001)
print(f"优化器后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

x = torch.rand((10240,30,18)).to(device)
y = torch.rand(10240).to(device)
print(f"数据后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

prediction = mymodel(x, x)
criteria = nn.MSELoss()
loss = criteria(prediction[0], y)
print(f"前向+损失后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

loss.backward()
print(f"反向后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

optimizer.step()
print(f"优化器step后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

optimizer.zero_grad()
print(f"zero_grad后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

x = torch.rand((1024,30,18)).to(device)
y = torch.rand(1024).to(device)
print(f"数据后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

prediction = mymodel(x, x)
criteria = nn.MSELoss()
loss = criteria(prediction[0], y)
print(f"前向+损失后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

loss.backward()
print(f"反向后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

optimizer.step()
print(f"优化器step后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")

optimizer.zero_grad()
print(f"zero_grad后: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
#
# x = [torch.rand(5,5),1,1]
# a = LambdaLayer(lambda x: x[0][:, -1])
# print(x[0].numpy(),a(x).numpy())
loss=nn.CrossEntropyLoss()
prediction1 = torch.tensor([[0.5, 0.5]])
prediction2 = torch.tensor([[1.0, 0]])
prediction3 = torch.tensor([[0, 1.0]])
prediction4 = torch.tensor([[1.0, 4.0]])
target = torch.tensor([0])
print(loss(prediction1, target), loss(prediction2, target), loss(prediction3, target), loss(prediction4, target))