import copy
import torch
import numpy as np
from system.client.clientbase import Client

class clientAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)


# 在训练过程中更新EMA



    def train(self):

        # self.model.to(self.device)
        ####模式
        self.model.train()

        if self.args.EDI_Freeze:
            for param in self.model.LHDR.parameters():
                param.requires_grad = False

        max_local_epochs = self.local_epochs

        epoch_train_losses = []
        epoch_test_losses_local = []
        epoch_test_losses_ema = []

        for epoch in range(max_local_epochs):
            print(f"global round {self.global_round} client{self.id}  local epoch: {epoch} ")
            global_step = (self.global_round * max_local_epochs + epoch) * len(self.trainloader)
            global_step_test = (self.global_round * max_local_epochs + epoch)

            total_train_loss = 0.0
            total_train_samples = 0

            self.model.train()
            for i, (x, y) in enumerate(self.trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                output, _, _ = self.model(x)
                loss = self.loss(output, y)
                self.writer.add_scalar('train/steploss_client'+str(self.id),torch.sqrt(loss),global_step+i)
                self.optimizer.zero_grad()
                loss.backward()
                if self.client_clip:
                    grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
                self.optimizer.step()

                total_train_loss += loss* len(x)
                total_train_samples += len(x)

            avg_epoch_train_loss = total_train_loss / total_train_samples
            epoch_train_losses.append(torch.sqrt(avg_epoch_train_loss).item())
            self.writer.add_scalar('train/epochloss_client' + str(self.id), torch.sqrt(avg_epoch_train_loss), global_step + i)

            self.model.eval()
            self.ema_model.eval()
            for i, (x, y) in enumerate(self.testloader):
                x = x.to(self.device)
                y = y.to(self.device)

                output_local, _, _ = self.model(x)
                output_ema, _, _ = self.ema_model(x)
                loss_local = self.loss(output_local, y)
                loss_ema = self.loss(output_ema, y)
                self.writer.add_scalar('test/client'+str(self.id),torch.sqrt(loss_local),global_step_test+i)
                self.writer.add_scalar('test/client_ema'+str(self.id),torch.sqrt(loss_ema),global_step_test+i)

            epoch_test_losses_local.append(torch.sqrt(loss_local).item())
            epoch_test_losses_ema.append(torch.sqrt(loss_ema).item())

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            if self.args.use_ema:
                self.update_ema()

        # avg_train_loss = np.mean(epoch_train_losses)
        # avg_test_loss_local = np.mean(epoch_test_losses_local)
        # avg_test_loss_ema = np.mean(epoch_test_losses_ema)
        # self.writer.add_scalar('train/round_avg_loss_client' + str(self.id), np.sqrt(avg_train_loss), self.global_round)
        # self.writer.add_scalar('test/round_avg_loss_client' + str(self.id), np.sqrt(avg_test_loss_local), self.global_round)
        # if self.args.use_ema:
        #     self.writer.add_scalar('test/round_avg_loss_ema_client' + str(self.id), np.sqrt(avg_test_loss_ema), self.global_round)

    def update_ema(self):
        for ema_param, param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                param.data, alpha=1 - self.ema_decay
            )
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
