import copy
import torch
import numpy as np
from system.client.clientbase import Client
from utils.metric import SF

class clientCADA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # self.optimizer_enc = torch.optim.Adam(list(self.model.F.parameters())+list(self.model.LHDR.parameters())+
        #                                       list(self.model.predictor.parameters()), lr=self.learning_rate)
        # self.optimizer_nce = torch.optim.Adam(self.InfoNCEHead.parameters(), lr=1e-2)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.F.parameters(), 'lr': self.learning_rate},
            {'params': self.model.LHDR.parameters(), 'lr': self.learning_rate},
            {'params': self.model.predictor.parameters(), 'lr': self.learning_rate},
            {'params': self.model.InfoNCEHead.parameters(), 'lr': 1e-2}
        ])

    def train(self):

        # self.model.to(self.device)
        ####模式
        self.model.train()

        if self.args.EDI_Freeze:
            for param in self.model.LHDR.parameters():
                param.requires_grad = False

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            print(f"global round {self.global_round} client{self.id}  local epoch: {epoch} ")
            global_step = (self.global_round * max_local_epochs + epoch) * len(self.trainloader)
            global_step_test = (self.global_round * max_local_epochs + epoch)

            total_train_loss = 0.0
            total_train_samples = 0

            for i, (x, y) in enumerate(self.trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, shallow, middle, infonce_loss = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/steploss_client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.writer.add_scalar('train/stepinfonceloss_client'+str(self.id),torch.sqrt(infonce_loss),global_step+i)
                if self.args.enable_CADA:
                    print("enable_CADA")
                    loss += self.args.info_lambda*infonce_loss
                self.optimizer.zero_grad()
                loss.backward()
                if self.client_clip:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
                self.optimizer.step()

                total_train_loss += loss* len(x)
                total_train_samples += len(x)

            avg_train_loss = total_train_loss / total_train_samples
            self.writer.add_scalar('train/epochloss_client' + str(self.id), torch.sqrt(avg_train_loss), global_step + i)

            for i, (x, y) in enumerate(self.testloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _, _, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss),global_step_test+i)

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

    def soft_update(self, model):
        if self.args.F_FedAvg:
            for new_param, old_param in zip(model.F.parameters(), self.model.F.parameters()):#可以选择
                old_param.data = self.miu_su * old_param.data + (1 - self.miu_su) * new_param.data
        if self.args.EDI_FedAvg:
            for new_param, old_param in zip(model.LHDR.parameters(), self.model.LHDR.parameters()):#可以选择
                old_param.data = self.miu_su * old_param.data + (1 - self.miu_su) * new_param.data
        if self.args.P_FedAvg:
            for new_param, old_param in zip(model.unique.parameters(), self.model.unique.parameters()):#可以选择
                old_param.data = self.miu_su * old_param.data + (1 - self.miu_su) * new_param.data

    def get_feature(self):
        self.shallow_feature=[]
        self.middle_feature=[]
        for i, (x, y) in enumerate(self.trainloader):
            x = x.to(self.device)
            y = y.to(self.device)
            _, shallow, middle, _ = self.model(x)
            self.shallow_feature.append(shallow.detach())
            self.middle_feature.append(middle.detach())

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
        test_num=x.size(0)

        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()


        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            output, shallow, middle, _ = self.model(x)
            loss= self.loss(output, y)
            score = SF(y, output)

        return loss, test_num,score