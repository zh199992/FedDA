import copy
import torch
import numpy as np
from system.client.clientbase import Client
#改optimizer
class clientGHDR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.learning_rate1 = float(args.local_learning_rate.split(',')[0])
        self.learning_rate2 = float(args.local_learning_rate.split(',')[1])
        self.optimizer1 = torch.optim.Adam(self.model.unique.parameters(), lr=self.learning_rate1)#如果是别的方案就不能提前设置
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)#如果是别的方案就不能提前设置
        self.learning_rate_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer1,
            gamma=args.learning_rate_decay_gamma
        )
        self.optimizer2 = torch.optim.Adam(self.model.LHDR.parameters(), lr=self.learning_rate2)#如果是别的方案就不能提前设置
        self.learning_rate_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer2,
            gamma=args.learning_rate_decay_gamma
        )
        self.local_epochs1 = int(args.local_epochs.split(',')[0])
        self.local_epochs2 = int(args.local_epochs.split(',')[1])


    def train(self):
        trainloader = self.load_train_data()
        testloader = self.load_test_data()
        # self.model.to(self.device)
        ####模式
        self.model.train()

        for epoch in range(self.local_epochs1):
            print(f"client{self.id}  local epoch: {epoch} ")
            global_step = (self.global_round * (self.local_epochs1+self.local_epochs2) + epoch) * len(trainloader)
            global_step_test = (self.global_round * (self.local_epochs1+self.local_epochs2) + epoch)

            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer1.zero_grad()
                loss.backward()
                if self.client_clip==True:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.unique.parameters(), max_norm=100)
                self.optimizer1.step()

            for i, (x, y) in enumerate(testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)

        if self.learning_rate_decay:
            self.learning_rate_scheduler1.step()



        for epoch in range(self.local_epochs2):
            print(f"client{self.id}  local epoch: {self.local_epochs1+epoch} ")
            global_step = (self.global_round * (self.local_epochs1+self.local_epochs2) + self.local_epochs1+epoch) * len(trainloader)
            global_step_test = (self.global_round * (self.local_epochs1+self.local_epochs2) +self.local_epochs1+ epoch)

            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer2.zero_grad()
                loss.backward()
                if self.client_clip==True:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.LHDR.parameters(), max_norm=100)
                self.optimizer2.step()

            for i, (x, y) in enumerate(testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)



        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler2.step()
    def set_parameters(self, model):
        for new_param, old_param in zip(model.LHDR.parameters(), self.model.LHDR.parameters()):
            old_param.data = new_param.data.clone()
        for new_param, old_param in zip(model.F.parameters(), self.model.F.parameters()):
            old_param.data = new_param.data.clone()
