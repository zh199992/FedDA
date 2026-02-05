import copy
import torch
import numpy as np
from system.client.clientbase import Client
from utils.data_utils import read_client_data_iid
from torch.utils.data import DataLoader


class clientAvgiid(Client):
    def __init__(self, args, id, trainset, testset, train_samples, test_samples, **kwargs):
        self.trainset = trainset
        self.testset = testset
        super().__init__(args, id, train_samples, test_samples, **kwargs)#即使在基类的 __init__ 中调用 self.load_train_data，实际调用的仍然是子类的 load_train_data 方法。

    def train(self):

        # self.model.to(self.device)
        ####模式
        self.model.train()

        if self.args.EDI_Freeze:
            for param in self.model.LHDR.parameters():
                param.requires_grad = False

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            print(f"client{self.id}  local epoch: {epoch} ")
            global_step = (self.global_round * max_local_epochs + epoch) * len(self.trainloader)
            global_step_test = (self.global_round * max_local_epochs + epoch)

            for i, (x, y) in enumerate(self.trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer.zero_grad()
                loss.backward()
                if self.client_clip:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
                self.optimizer.step()

            for i, (x, y) in enumerate(self.testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)

        # self.model.cpu()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()


    def set_parameters(self, model):
        if self.args.F_FedAvg:
            for new_param, old_param in zip(model.F.parameters(), self.model.F.parameters()):#可以选择
                old_param.data = new_param.data.clone()
        if self.args.EDI_FedAvg:
            for new_param, old_param in zip(model.LHDR.parameters(), self.model.LHDR.parameters()):#可以选择
                old_param.data = new_param.data.clone()
        if self.args.P_FedAvg:
            for new_param, old_param in zip(model.unique.parameters(), self.model.unique.parameters()):#可以选择
                old_param.data = new_param.data.clone()

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.trainset, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.testset, batch_size, drop_last=False, shuffle=True)

