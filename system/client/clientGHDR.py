import copy
import torch
import numpy as np
from system.client.clientbase import Client
import adamod

#改optimizer
class clientGHDR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.learning_rate1 = args.local_learning_rate
        self.learning_rate2 = args.local_learning_rate
        if self.args.optimizer_client == 'adamod':
            self.optimizer1 = adamod.AdaMod(self.model.unique.parameters(), lr=self.learning_rate1)
        elif self.args.optimizer_client == 'adam':
            self.optimizer1 = torch.optim.Adam(self.model.unique.parameters(), lr=self.learning_rate1)
        elif self.args.optimizer_client == 'sgd':
            self.optimizer1 = torch.optim.SGD(self.model.unique.parameters(), lr=self.learning_rate1)
        else:
            raise NotImplementedError
        self.learning_rate_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer1,
            gamma=args.learning_rate_decay_gamma
        )
        if self.args.optimizer_client == 'adamod':
            self.optimizer2 = adamod.AdaMod(list(self.model.F.parameters()) + list(self.model.LHDR.parameters()), lr=self.learning_rate2)
        elif self.args.optimizer_client == 'adam':
            self.optimizer2 = torch.optim.Adam(list(self.model.F.parameters()) + list(self.model.LHDR.parameters()), lr=self.learning_rate2)
        elif self.args.optimizer_client == 'sgd':
            self.optimizer2 = torch.optim.SGD(list(self.model.F.parameters()) + list(self.model.LHDR.parameters()), lr=self.learning_rate2)
        else:
            raise NotImplementedError
        self.learning_rate_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer2,
            gamma=args.learning_rate_decay_gamma
        )
        self.local_epochs1 = args.local_epochs
        self.local_epochs2 = 50-args.local_epochs


    def train(self):
        # self.model.to(self.device)
        ####模式
        self.model.train()

        for epoch in range(self.local_epochs1):
            print(f"global round {self.global_round} client{self.id}  local epoch: {epoch} ")
            global_step = (self.global_round * (self.local_epochs1+self.local_epochs2) + epoch) * len(self.trainloader)
            global_step_test = (self.global_round * (self.local_epochs1+self.local_epochs2) + epoch)

            for i, (x, y) in enumerate(self.trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, shallow, middle = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer1.zero_grad()
                loss.backward()
                if self.client_clip==True:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.unique.parameters(), max_norm=100)
                self.optimizer1.step()
            if self.learning_rate_decay:
                self.learning_rate_scheduler1.step()
            print(self.learning_rate_scheduler1.state_dict())

            for i, (x, y) in enumerate(self.testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, shallow, middle = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)




        for epoch in range(self.local_epochs2):
            print(f"client{self.id}  local epoch: {self.local_epochs1+epoch} ")
            global_step = (self.global_round * (self.local_epochs1+self.local_epochs2) + self.local_epochs1+epoch) * len(self.trainloader)
            global_step_test = (self.global_round * (self.local_epochs1+self.local_epochs2) +self.local_epochs1+ epoch)

            for i, (x, y) in enumerate(self.trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, shallow, middle = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer2.zero_grad()
                self.optimizer1.zero_grad()
                loss.backward()
                if self.client_clip==True:
                    backbone_params = list(self.model.F.parameters()) + list(self.model.LHDR.parameters())
                    torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=100)
                self.optimizer2.step()

            for i, (x, y) in enumerate(self.testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, shallow, middle = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)

        # self.model.cpu()
            if self.learning_rate_decay:
                self.learning_rate_scheduler2.step()
            print(self.learning_rate_scheduler2.state_dict())


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

