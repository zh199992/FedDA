import copy
import torch
import numpy as np
from system.client.clientbase import Client
#改变 1.optimizer和scheduler 2.在train时存储shallow_feature 3.set_parameters (决定了F是否参与聚合)
class clientDA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # if self.args.algorithm == 'FedAvg':
        # params_to_optimize = list(self.model.E_DS.parameters()) + list(self.model.p_pre.parameters())
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)#如果是别的方案就不能提前设置
        # self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=args.learning_rate_decay_gamma
        # )


    def train(self):
        ####模式
        self.model.train()

        if self.args.EDI_Freeze:
            for param in self.model.LHDR.parameters():
                param.requires_grad = False

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            print(f"global round {self.global_round} client{self.id}  local epoch: {epoch} ")
            global_step = (self.global_round * (max_local_epochs) + epoch) * len(self.trainloader)
            global_step_test = (self.global_round * (max_local_epochs) + epoch)

            for i, (x, y) in enumerate(self.trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer.zero_grad()
                loss.backward()
                if self.client_clip==True:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
                self.optimizer.step()

            for i, (x, y) in enumerate(self.testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output,_ ,_ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)

        # self.model.cpu()

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

        #
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

