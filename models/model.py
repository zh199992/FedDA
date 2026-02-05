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
        self.block=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, padding='same'),
            Mish()
        )
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

class GHDR_FL_new(nn.Module):#
    def __init__(self,input_size,conv_init='kaiming_uniform', gru_init=None, linear_init='xavier_uniform'):
        super(GHDR_FL_new, self).__init__()
        self.mode='phase1'
        self.input_size=input_size
        self.filter_num=10
        self.filter_length=10
        self.F=nn.Sequential(
            conv_block(1,self.filter_num,kernel_size=(self.filter_length,input_size)),
            conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,input_size)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, input_size))
        )
        self.LHDR=nn.Sequential(
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, input_size)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, input_size)),
            conv_block(self.filter_num, 1, kernel_size=(3, input_size))
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
        feature_reversed = GradReverse.grad_reverse(feature, constant)

        return self.discriminator(feature_reversed), feature

class Cloud_GHDR_new(nn.Module):
    def __init__(self,input_size, window_size, client_number, conv_init='kaiming_uniform',  linear_init='xavier_uniform'):
        super(Cloud_GHDR_new, self).__init__()
        self.client_number=client_number
        self.input_size=input_size
        self.filter_num=10
        self.filter_length=10
        self.F=nn.Sequential(    #只用来保存
            conv_block(1,self.filter_num,kernel_size=(self.filter_length,input_size)),
            conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,input_size)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, input_size))
        )

        self.LHDR=nn.Sequential(
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, input_size)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, input_size)),
            conv_block(self.filter_num, 1, kernel_size=(3, input_size)),
        )

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
        feature_reversed = GradReverse.grad_reverse(feature, constant)

        return self.discriminator(feature_reversed), feature

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
            nn.Dropout(0.5),#不知道具體
            nn.Linear(30,100),
            nn.Tanh(),
            nn.Linear(100,1)
        )
    def forward(self, x):
        # x=x.permute((0,2,1))
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

class conv_DANN(nn.Module):#卷积层 (nn.Conv2d)：默认使用Kaiming均匀分布初始化（He初始化）。
#循环神经网络层 (nn.GRU)：默认使用Xavier均匀分布初始化。
#全连接层 (nn.Linear)：默认使用Kaiming均匀分布初始化（He初始化），并且偏置项根据输入特征数量计算一个适当的边界值进行均匀分布初始化。
    def __init__(self,input_size,conv_init='kaiming_uniform', gru_init=None, linear_init='xavier_uniform'):
        super(conv_DANN, self).__init__()
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
        self.discriminator = nn.Sequential(
            # nn.Flatten(),
                            nn.Linear(30, 700),
            # nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(700, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
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
    def forward(self, source, target, constant=1):
        source=self.F(source.unsqueeze(1))#(b,1,30,18)->>(b,10,30,18)
        target=self.F(target.unsqueeze(1))
        source_feature=self.LHDR(source).squeeze(1)#(b,10,30,18)->>(b,30,18)
        target_feature=self.LHDR(target).squeeze(1)
        prediction = self.unique(source_feature)#LSTM中会有空间相关性的交互
        source_feature = torch.mean(source_feature, dim=2)#特征维度平均#(b,30,18)->>(b,30)
        target_feature = torch.mean(target_feature, dim=2)#特征维度平均
        source_feature = GradReverse.grad_reverse(source_feature, constant)
        target_feature = GradReverse.grad_reverse(target_feature, constant)

        return (prediction, self.discriminator(source_feature), self.discriminator(target_feature),
            source_feature, target_feature)

class conv_DANN2(nn.Module):#卷积层 (nn.Conv2d)：默认使用Kaiming均匀分布初始化（He初始化）。
#循环神经网络层 (nn.GRU)：默认使用Xavier均匀分布初始化。
#全连接层 (nn.Linear)：默认使用Kaiming均匀分布初始化（He初始化），并且偏置项根据输入特征数量计算一个适当的边界值进行均匀分布初始化。
    def __init__(self,input_size,conv_init='kaiming_uniform', gru_init=None, linear_init='xavier_uniform'):
        super(conv_DANN2, self).__init__()
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
        self.discriminator = nn.Sequential(
            nn.Flatten(),
                            nn.Linear(540, 700),
            # nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(700, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
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
    def forward(self, source, target, constant=1):
        source=self.F(source.unsqueeze(1))#(b,1,30,18)->>(b,10,30,18)
        target=self.F(target.unsqueeze(1))
        source_feature=self.LHDR(source).squeeze(1)#(b,10,30,18)->>(b,30,18)
        target_feature=self.LHDR(target).squeeze(1)
        prediction = self.unique(source_feature)#LSTM中会有空间相关性的交互
        # source_feature = torch.mean(source_feature, dim=2)#特征维度平均#(b,30,18)->>(b,30)
        # target_feature = torch.mean(target_feature, dim=2)#特征维度平均
        source_feature = GradReverse.grad_reverse(source_feature, constant)
        target_feature = GradReverse.grad_reverse(target_feature, constant)

        return (prediction, self.discriminator(source_feature), self.discriminator(target_feature),
            torch.mean(source_feature, dim=2), torch.mean(target_feature, dim=2))


class FedCADA(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=32, num_layers=5, seq_len=30):
        super(FedCADA,self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.F = nn.LSTM(input_dim, hidden_dim, num_layers - 3, batch_first=True)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
        #                     batch_first=True, bidirectional=True)
        self.LHDR = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )
        self.InfoNCEHead = nn.ModuleList([
            nn.Linear(hidden_dim, input_dim) for _ in range(seq_len)
        ])  # 输入(n,30,14) 输出(30,n,30,14),如果第k个head，得到的正样本是(k,n,k,14),负样本是(k,n,[:k],14)+(k,n,[k+1:],14)

    def forward(self, x):
        shallow, _ = self.F(x)
        f_T, _ = self.LHDR(shallow)
        y_pred = self.predictor(f_T[:, -1, :])  # [B, hidden*2]

        B, K, M = x.shape
        assert K == self.seq_len and M == self.input_dim
        total_loss = 0.0
        for k in range(self.seq_len):
            q_k = self.InfoNCEHead[k](f_T[:, -1, :])  # [B, input_dim]
            x_k = x[:, k, :]  # [B, input_dim]

            # 构造相似度：q_k 与 同样本所有 x_j 的点积（j=0,...,K-1）
            # 正样本：j == k
            # 负样本：j != k
            # 计算 q_k 与 X_T 中所有时间步的点积
            # sim[i, j] = q_k[i] · x_j[i]  （注意：只与自己样本的 x_j 比较）
            # 所以不能用 mm(q_k, X_T[:, j].T)，因为那是跨样本

            # 正确做法：逐样本计算（或使用广播）
            # q_k: [B, M], X_T: [B, K, M] → 点积在 M 上，结果 [B, K]
            # print(q_k.size(), x.size())
            sim = torch.einsum('bm,bkm->bk', q_k, x)  # [B, K] 每个样本在第k步的transformed feature和原始序列每一步的sim
            # sim = sim / 0.1  # temperature

            # 标签：每个样本的正样本在位置 k
            labels = torch.full((B,), k, device=f_T.device, dtype=torch.long)

            loss_k = F.cross_entropy(sim, labels)
            total_loss += loss_k

        return y_pred, shallow, f_T[:, -1, :],total_loss / K

class Cloud_FedCADA(nn.Module):
    def __init__(self, client_number, input_dim=14, hidden_dim=32, num_layers=5, seq_len=30):
        super(Cloud_FedCADA, self).__init__()
        self.client_number=client_number
        self.input_dim=input_dim
        self.F = nn.LSTM(input_dim, hidden_dim, num_layers - 3, batch_first=True)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
        #                     batch_first=True, bidirectional=True)
        # self.LHDR = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True)
        self.LHDR = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.client_number)
            )
    def forward(self, shallow, constant=1):
        feature, _=self.LHDR(shallow)
        feature_reversed = GradReverse.grad_reverse(feature[:,-1,:], constant)

        return self.discriminator(feature_reversed), feature[:,-1,:]
class Cloud_FedCADA_newda(nn.Module):
    def __init__(self, client_number, input_dim=14, hidden_dim=32, num_layers=5, seq_len=30):
        super(Cloud_FedCADA_newda, self).__init__()
        self.client_number=client_number
        self.input_dim=input_dim
        self.F = nn.LSTM(input_dim, hidden_dim, num_layers - 3, batch_first=True)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
        #                     batch_first=True, bidirectional=True)
        # self.LHDR = nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True)
        self.LHDR_list = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, 3, batch_first=True)
            for _ in range(client_number)
        ])
        self.discriminator_list =nn.ModuleList([
            nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            )
            for _ in range(client_number)
        ])
    def forward(self, shallow, source_middle, domain_ids=None, constant=1):
        feature, _=self.LHDR_list[domain_ids](shallow)
        feature_reversed = GradReverse.grad_reverse(feature[:,-1,:], constant)
        discriminator_input = torch.cat([feature_reversed, source_middle])
        return self.discriminator_list[domain_ids](discriminator_input), feature[:,-1,:]

class CADA(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=32, num_layers=3, seq_len=30, lambda_nce=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lambda_nce = lambda_nce
        self.seq_len = seq_len

        # Shared BiLSTM encoder (used for both source and target)
        self.encoder_learning = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.encoder_frozen = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )


        # RUL predictor (trained on source)
        self.rul_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

        # Domain discriminator (for adversarial loss)
        self.domain_disc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # InfoNCE head: maps f_T → q_k ∈ R^M for each timestep k
        # We use a single linear layer shared across timesteps (as in many CPC variants)
        self.nce_head = nn.ModuleList([nn.Linear(hidden_dim * 2, input_dim) for _ in range(seq_len)])

    def forward(self, x_learning, x_frozen):#应该输出
        f_learning, _ = self.encoder_learning(x_learning)#预训练时，
        f_frozen, _ = self.encoder_frozen(x_frozen)
        y_pred = self.rul_predictor(f_learning[:, -1, :])  # [B, hidden*2]

        B, K, M = x_learning.shape
        assert K == self.seq_len and M == self.input_dim
        total_loss = 0.0
        for k in range(self.seq_len):
            q_k = self.nce_head[k](f_learning[:, -1, :])  # [B, input_dim]
            x_k = x_learning[:, k, :]  # [B, input_dim]

            # 构造相似度：q_k 与 同样本所有 x_j 的点积（j=0,...,K-1）
            # 正样本：j == k
            # 负样本：j != k
            # 计算 q_k 与 X_T 中所有时间步的点积
            # sim[i, j] = q_k[i] · x_j[i]  （注意：只与自己样本的 x_j 比较）
            # 所以不能用 mm(q_k, X_T[:, j].T)，因为那是跨样本

            # 正确做法：逐样本计算（或使用广播）
            # q_k: [B, M], X_T: [B, K, M] → 点积在 M 上，结果 [B, K]
            # print(q_k.size(), x.size())
            sim = torch.einsum('bm,bkm->bk', q_k, x_learning)  # [B, K] 每个样本在第k步的transformed feature和原始序列每一步的sim
            # sim = sim / 0.1  # temperature

            # 标签：每个样本的正样本在位置 k
            labels = torch.full((B,), k, device=f_learning.device, dtype=torch.long)

            loss_k = F.cross_entropy(sim, labels)
            total_loss += loss_k
        f_learning = GradReverse.grad_reverse(f_learning[:,-1,:], 1)
        f_frozen = GradReverse.grad_reverse(f_frozen[:,-1,:], 1)
        domain_label_learning = self.domain_disc(f_learning)
        # domain_label_learning = torch.sigmoid(self.domain_disc(f_learning))
        domain_label_frozen = self.domain_disc(f_frozen)
        # domain_label_frozen = torch.sigmoid(self.domain_disc(f_frozen))
        loss_d = -(torch.mean(torch.log(domain_label_frozen + 1e-8)) +             #source domain的label=1
                torch.mean(torch.log(1 - domain_label_learning + 1e-8))
        )
        InfoNCE_loss = total_loss / K
        return y_pred, f_learning, f_frozen, loss_d, InfoNCE_loss #要改变mmd计算方法可以在mmd输入时计算
        # return y_pred, domain_label_learning, domain_label_frozen, f_learning, f_frozen, InfoNCE_loss #要改变mmd计算方法可以在mmd输入时计算