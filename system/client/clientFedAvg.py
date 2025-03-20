import copy
import torch
import numpy as np
from system.client.clientbase import Client

class clientAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        testloader = self.load_test_data()
        # self.model.to(self.device)
        ####模式
        self.model.train()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            print(f"client{self.id}  local epoch: {epoch} ")
            global_step = (self.global_round * (max_local_epochs) + epoch) * len(trainloader)
            global_step_test = (self.global_round * (max_local_epochs) + epoch)

            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer.zero_grad()
                loss.backward()
                if self.client_clip==True:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
                self.optimizer.step()

            for i, (x, y) in enumerate(testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

