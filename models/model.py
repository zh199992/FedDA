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

# model
hidden_size1=64
hidden_size2=64

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(conv_block, self).__init__()
        self.block=nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding='same'),
                                 Mish())
    def forward(self, x):
        return self.block(x)

class GHDR_FL(nn.Module):#卷积层 (nn.Conv2d)：默认使用Kaiming均匀分布初始化（He初始化）。
#循环神经网络层 (nn.GRU)：默认使用Xavier均匀分布初始化。
#全连接层 (nn.Linear)：默认使用Kaiming均匀分布初始化（He初始化），并且偏置项根据输入特征数量计算一个适当的边界值进行均匀分布初始化。
    def __init__(self,input_size,conv_init='kaiming_uniform', gru_init=None, linear_init='xavier_uniform'):
        super(GHDR_FL, self).__init__()
        self.mode='phase1'
        self.input_size=input_size
        self.filter_num=10
        self.filter_length=10
        self.F=nn.Sequential(
            conv_block(1,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1))
        )
        self.LHDR=nn.Sequential(
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, 1, kernel_size=(3, 1))
        )
        # self.unique=nn.Sequential(
        #     nn.GRU(input_size=self.input_size,hidden_size=50,num_layers=1,batch_first=True),
        #     LambdaLayer(lambda x: x[0][:, -1, :]),
        #     nn.Linear(50,700),
        #     # nn.Tanh(),
        #     Mish(),
        #     nn.Linear(700,200),
        #     # nn.Tanh(),
        #     Mish(),
        #     nn.Linear(200, 1),
        #
        # )
        self.unique = nn.Sequential(
            nn.GRU(input_size=self.input_size, hidden_size=50, num_layers=1, batch_first=True),
            LambdaLayer(lambda x: x[0][:, -1, :]),
            nn.Dropout(0.3),
            nn.Linear(50, 700),
            # nn.Tanh(),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(700, 200),
            # nn.Tanh(),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(200, 1),
        )
        conv_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if conv_init in conv_initializers:
            for module in self.LHDR.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)
            for module in self.F.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)

        # Initialize GRU layers (if specified)
        if gru_init is not None:
            for name, param in self.unique[0].named_parameters():
                if 'weight' in name:
                    if gru_init == 'xavier_uniform':
                        init.xavier_uniform_(param)
                    elif gru_init == 'xavier_normal':
                        init.xavier_normal_(param)
                    elif gru_init == 'kaiming_uniform':
                        init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'kaiming_normal':
                        init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'normal':
                        init.normal_(param, mean=0.0, std=0.02)
                    elif gru_init == 'uniform':
                        init.uniform_(param, a=-0.1, b=0.1)
                elif 'bias' in name:
                    init.constant_(param, 0)

        # Initialize Linear layers
        linear_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if linear_init in linear_initializers:
            for module in self.unique.modules():
                if isinstance(module, nn.Linear):
                    linear_initializers[linear_init](module.weight)
                    init.constant_(module.bias, 0)
    def forward(self, input):
        input=input.unsqueeze(1)
        shallow=self.F(input)
        middle=self.LHDR(shallow)
        middle=middle.squeeze(1)
        output=self.unique(middle)
        return output, shallow, middle

class GHDR_test(nn.Module):#卷积层 (nn.Conv2d)：默认使用Kaiming均匀分布初始化（He初始化）。
#循环神经网络层 (nn.GRU)：默认使用Xavier均匀分布初始化。
#全连接层 (nn.Linear)：默认使用Kaiming均匀分布初始化（He初始化），并且偏置项根据输入特征数量计算一个适当的边界值进行均匀分布初始化。
    def __init__(self,input_size,conv_init='kaiming_uniform', gru_init=None, linear_init='xavier_uniform'):
        super(GHDR_test, self).__init__()
        self.mode='phase1'
        self.input_size=input_size
        self.filter_num=10
        self.filter_length=10
        self.F=nn.Sequential(
            conv_block(1,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1))
        )
        self.LHDR=nn.Sequential(
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, 1, kernel_size=(3, 1))
        )
        self.GRU=nn.GRU(input_size=self.input_size,hidden_size=50,num_layers=1,batch_first=True)

        self.unique=nn.Sequential(
            nn.Linear(50,700),
            # nn.Tanh(),
            Mish(),
            nn.Linear(700,200),
            # nn.Tanh(),
            Mish(),
            nn.Linear(200, 1),

        )
        conv_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if conv_init in conv_initializers:
            for module in self.LHDR.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)
            for module in self.F.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)

        # Initialize GRU layers (if specified)
        if gru_init is not None:
            for name, param in self.unique[0].named_parameters():
                if 'weight' in name:
                    if gru_init == 'xavier_uniform':
                        init.xavier_uniform_(param)
                    elif gru_init == 'xavier_normal':
                        init.xavier_normal_(param)
                    elif gru_init == 'kaiming_uniform':
                        init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'kaiming_normal':
                        init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'normal':
                        init.normal_(param, mean=0.0, std=0.02)
                    elif gru_init == 'uniform':
                        init.uniform_(param, a=-0.1, b=0.1)
                elif 'bias' in name:
                    init.constant_(param, 0)

        # Initialize Linear layers
        linear_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if linear_init in linear_initializers:
            for module in self.unique.modules():
                if isinstance(module, nn.Linear):
                    linear_initializers[linear_init](module.weight)
                    init.constant_(module.bias, 0)
    def forward(self, input):
        input=input.unsqueeze(1)
        shallow=self.F(input)
        middle=self.LHDR(shallow)
        middle=self.GRU(middle.squeeze(1))[0][:, -1, :]
        output=self.unique(middle)
        return output, shallow, middle

class GHDR_FL_testeds(nn.Module):
    def __init__(self,input_size,conv_init='kaiming_uniform', gru_init=None, linear_init='xavier_uniform'):
        super(GHDR_FL_testeds, self).__init__()
        self.mode='phase1'
        self.input_size=input_size
        self.filter_num=10
        self.filter_length=10
        self.F=nn.Sequential(
            conv_block(1,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1))
        )
        self.LHDR=nn.Sequential(
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, 1, kernel_size=(3, 1))
        )
        self.E_DS = nn.Sequential(
            # conv_block(1,self.filter_num,kernel_size=(self.filter_length,1)),
            # conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,1)),
            # conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, 1, kernel_size=(3, 1))
        )
        self.unique=nn.Sequential(
            nn.GRU(input_size=2*self.input_size,hidden_size=50,num_layers=1,batch_first=True),
            LambdaLayer(lambda x: x[0][:, -1, :]),
            nn.Linear(50,700),
            # nn.Tanh(),
            Mish(),
            nn.Linear(700,200),
            # nn.Tanh(),
            Mish(),
            nn.Linear(200, 1),

        )
        conv_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if conv_init in conv_initializers:
            for module in self.LHDR.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)
            for module in self.F.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)
            for module in self.E_DS.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)

        # Initialize GRU layers (if specified)
        if gru_init is not None:
            for name, param in self.unique[0].named_parameters():
                if 'weight' in name:
                    if gru_init == 'xavier_uniform':
                        init.xavier_uniform_(param)
                    elif gru_init == 'xavier_normal':
                        init.xavier_normal_(param)
                    elif gru_init == 'kaiming_uniform':
                        init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'kaiming_normal':
                        init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'normal':
                        init.normal_(param, mean=0.0, std=0.02)
                    elif gru_init == 'uniform':
                        init.uniform_(param, a=-0.1, b=0.1)
                elif 'bias' in name:
                    init.constant_(param, 0)

        # Initialize Linear layers
        linear_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if linear_init in linear_initializers:
            for module in self.unique.modules():
                if isinstance(module, nn.Linear):
                    linear_initializers[linear_init](module.weight)
                    init.constant_(module.bias, 0)
    def forward(self, input):
        input=input.unsqueeze(1)
        shallow=self.F(input)
        middle=torch.cat((self.LHDR(shallow), self.E_DS(shallow)), dim=-1)
        middle=middle.squeeze(1)
        output=self.unique(middle)
        return output, shallow, middle

class Cloud_GHDR(nn.Module):
    def __init__(self,input_size, window_size, client_number, conv_init='kaiming_uniform',  linear_init='xavier_uniform'):
        super(Cloud_GHDR, self).__init__()
        self.client_number=client_number
        self.input_size=input_size
        self.filter_num=10
        self.filter_length=10
        self.F=nn.Sequential(    #只用来保存
            conv_block(1,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1))
        )

        self.LHDR=nn.Sequential(
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, 1, kernel_size=(3, 1)),
        )
        # self.flat=nn.Flatten()
        # self.discriminator = nn.Sequential(
        #     # nn.Flatten(),
        #     nn.Linear(input_size*window_size, 700),
        #     nn.ReLU(),
        #     nn.Linear(700, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, self.client_number)
        # )
        self.discriminator = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(window_size, 700),
            # nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(700, 200),
            nn.ReLU(),
            nn.Linear(200, self.client_number)
        )
        conv_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if conv_init in conv_initializers:
            for module in self.LHDR.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)
        # Initialize Linear layers
        linear_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if linear_init in linear_initializers:
            for module in self.discriminator.modules():
                if isinstance(module, nn.Linear):
                    linear_initializers[linear_init](module.weight)
                    init.constant_(module.bias, 0)
    def forward(self, x, constant=1):
        feature=self.LHDR(x)
        # feature=self.flat(feature)

        feature=feature.squeeze(1)
        # feature = torch.mean(feature, dim=1)#时间维度平均
        feature = torch.mean(feature, dim=2)#特征维度平均
        feature = GradReverse.grad_reverse(feature, constant)

        return self.discriminator(feature), feature

class deepCNN(nn.Module):
    def __init__(self,input_size):
        super(deepCNN,self).__init__()
        self.model=nn.Sequential(
            nn.Conv1d(input_size, 10, 10, 1,padding='same'),
            nn.Tanh(),
            nn.Conv1d(10, 10, 10, 1, padding='same'),
            nn.Tanh(),
            nn.Conv1d(10, 10, 10, 1, padding='same'),
            nn.Tanh(),
            nn.Conv1d(10, 10, 10, 1, padding='same'),
            nn.Tanh(),
            nn.Conv1d(10, 1, 3, 1, padding='same'),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(30,100),
            nn.Tanh(),
            nn.Linear(100,1)
        )
    def forward(self, x):
        x=x.permute((0,2,1))
        return self.model(x)

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x