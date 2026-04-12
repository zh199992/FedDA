import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_client_data, read_client_data_centralized
from utils.metric import SF
import adamod


# train是在client类里重写 其他方法在clientbase类里
class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, writer, **kwargs):
        # torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.miu_su = args.miu_su
        self.train_samples = train_samples  # 用来干什么？计算RMSE吗？
        self.test_samples = test_samples
        self.batch_size = args.batch_size_client
        # self.learning_rate = float(args.local_learning_rate.split(',')[0])
        self.learning_rate = args.local_learning_rate
        # self.local_epochs = int(args.local_epochs.split(',')[0])
        self.local_epochs = int(args.local_epochs)
        self.client_schedule = args.client_schedule
        self.client_clip = args.client_clip

        self.loss = nn.MSELoss()
        if self.args.optimizer_client == 'adamod':
            self.optimizer = adamod.AdaMod(self.model.parameters(), lr=self.learning_rate)
        elif self.args.optimizer_client == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.args.optimizer_client == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.client_lr_decay
        self.writer = writer
        self.global_round = None
        self.trainloader = self.load_train_data()
        self.testloader = self.load_test_data(1024)

    def get_feature(self):#server.train 调用
        self.shallow_feature = []
        self.middle_feature = []
        self.label = []
        for i, (x, y) in enumerate(self.trainloader):
            x = x.to(self.device)
            y = y.to(self.device)
            _, shallow, middle = self.model(x)
            self.shallow_feature.append(shallow.detach())
            self.middle_feature.append(middle.detach())
            self.label.append(y.detach())

    def set_parameters(self, model):#server.send 调用
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if getattr(self.args, 'sub_algorithm', None) is None:
            train_data = read_client_data(self.dataset, self.id, self.args, is_train=True,
                                      train_ratio=self.args.train_ratio)
        else:
            train_data = read_client_data_centralized(self.dataset, self.args, is_train=True,
                                      train_ratio=self.args.train_ratio)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, self.args, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def test_metrics(self):
        # test_data = read_client_data(self.dataset, self.id, self.args, is_train=False)
        # x,y=test_data.data_tensor,test_data.target_tensor
        # test_num=len(test_data)
        x_list = []
        y_list = []

        for x_batch, y_batch in self.testloader:
            x_list.append(x_batch)
            y_list.append(y_batch)

        # 将所有批次的数据拼接成一个张量
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        test_num = x.size(0)

        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            output, shallow, middle = self.model(x)
            loss = self.loss(output, y)
            score = SF(y, output)

        return loss, test_num, score

    def train_metrics(self):  # 为什么需要这个
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
                output, _, _ = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
