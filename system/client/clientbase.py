import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from utils.metric import SF
#train是在client类里重写 其他方法在clientbase类里
class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, writer,**kwargs):
        # torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer

        self.train_samples = train_samples#用来干什么？计算RMSE吗？
        self.test_samples = test_samples
        self.batch_size = args.batch_size_client
        self.learning_rate = float(args.local_learning_rate.split(',')[0])
        self.local_epochs = int(args.local_epochs.split(',')[0])
        self.client_schedule=args.client_schedule
        self.client_clip=args.client_clip

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)#如果是别的方案就不能提前设置
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)#如果是别的方案就不能提前设置
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.writer = writer
        self.global_round=None

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, self.args, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, self.args, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def test_metrics(self):
        test_data = read_client_data(self.dataset, self.id, self.args, is_train=False)
        x,y=test_data.data_tensor,test_data.target_tensor
        test_num=len(test_data)
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()


        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            output, _ = self.model(x)
            loss= self.loss(output, y)
            score = SF(y, output)

        return loss, test_num,score

    def train_metrics(self):#为什么需要这个
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num