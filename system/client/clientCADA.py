import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_client_data, read_client_data_centralized
from utils.metric import SF
import adamod
from system.client.clientbase import Client
import sys
from utils.mmdloss import mmd_rbf


class clientCADA(Client):
    def __init__(self, args, id, train_samples, test_samples, target_id, **kwargs):
        self.source_id = id  # integer
        self.target_id = target_id
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.prediction_loss = nn.MSELoss()
        self.adv_loss = nn.CrossEntropyLoss()
        self.source_trainloader, self.target_train_loader = self.load_train_data()
        self.source_testloader, self.target_testloader = self.load_test_data(1024)
        self.all_trainset = read_client_data_centralized(self.dataset, self.args)
        self.all_testset = read_client_data_centralized(self.dataset, self.args, False)
        self.all_trainloader = DataLoader(self.all_trainset, self.batch_size, drop_last=False, shuffle=True)
        self.all_testloader = DataLoader(self.all_testset, self.batch_size, drop_last=False, shuffle=True)
        self.mode = args.mode
        if self.mode == "centralized":
            self.source_trainloader = self.all_trainloader
            self.source_testloader = self.all_testloader
            self.source_id = "all"
        # -----------------------------------
        self.early_stop = getattr(args, 'early_stop', False)  # 是否启用早停
        self.pretrain_early_stop = getattr(args, 'pretrain_early_stop', False)
        self.patience = 5  # 容忍多少轮不提升
        self.counter = 0  # 计数器
        self.early_stop_flag = False
        self.best_source_test_loss = 999
        self.best_target_test_loss = 999
        self.best_encoder_learning = None
        self.gamma = args.gamma
        self.model_backup = copy.deepcopy(self.model)
        self.per_layer = None
        self.lambda_nce = args.lambda_nce
        self.optimizer = torch.optim.Adam([
            {'params': self.model.encoder_learning.parameters(), 'lr': self.learning_rate},
            {'params': self.model.rul_predictor.parameters(), 'lr': self.learning_rate},
            {'params': self.model.domain_disc.parameters(), 'lr': self.learning_rate},
            {'params': self.model.nce_head.parameters(), 'lr': 1e-2}
        ])

    def train(self):
        print(f"[Client {self.source_id}] Source train data stats:")
        for i, (x1, y1) in enumerate(self.source_trainloader):
            print(f"  Batch {i}: x1.shape={x1.shape}, x1.mean={x1.mean().item():.4f}, y1.mean={y1.mean().item():.4f}")
            if i >= 2: break
        # self.model.to(self.device)
        ####模式
        self.model.train()

        max_local_epochs = self.local_epochs
        #--------------------------------------预训练ES
        for epoch in range(max_local_epochs):
            self.model.eval()
            for i, (data1, data2) in enumerate(zip(self.source_testloader, self.target_testloader)):
                x1, y1 = data1
                x2, y2 = data2
                x1, x2, y1, y2 = x1.to(self.device), x2.to(self.device), y1.to(self.device), y2.to(self.device)
                src_pred, _, _, _, _ = self.model(x1, x2)#第二个输入无用，只用第一个输入训练encoder_learning
                tgt_pred, _, _, _, _ = self.model(x2, x2)
                src_test_loss = self.prediction_loss(src_pred, y1)
                tgt_test_loss = self.prediction_loss(tgt_pred, y2)
                global_step_test = (self.global_round * max_local_epochs + epoch)
                # self.writer.add_scalar(f'pretrain-test/mmd{self.source_id}-{self.target_id}',
                #                        mmd_rbf(src_feature, tgt_feature), global_step_test)
                self.writer.add_scalar('pretrain-test/source_id' + str(self.source_id), torch.sqrt(src_test_loss),
                                       global_step_test)
                self.writer.add_scalar('pretrain-test/target_id' + str(self.target_id), torch.sqrt(tgt_test_loss),
                                       global_step_test)
                print(f"client{self.id}  local epoch: {epoch} ")

            if self.early_stop:
                # 假设 evaluate() 会把结果存到 self.rs_test_rmse 列表里
                # if src_test_loss < self.best_source_test_loss:
                #     self.best_source_test_loss = src_test_loss
                #     self.counter=0
                # else:
                #     self.counter += 1
                if self.mode == "Source-Only":
                    if torch.sqrt(tgt_test_loss) < self.best_target_test_loss:#以sqrt(tgt_test_loss)作为判断标准，
                        self.best_target_test_loss = torch.sqrt(tgt_test_loss)
                        self.best_encoder_learning = copy.deepcopy(self.model.encoder_learning)
                    #     self.counter = 0
                    # else:
                    #     self.counter += 1
                else:
                    if torch.sqrt(src_test_loss) < self.best_source_test_loss:  # 以sqrt(tgt_test_loss)作为判断标准，
                        self.best_source_test_loss = torch.sqrt(src_test_loss)
                        self.best_encoder_learning = copy.deepcopy(self.model.encoder_learning)
            #             self.counter = 0
            #         else:
            #             self.counter += 1
            #
            #     if self.counter >= self.patience:
            #         print("🔥🔥🔥 Early stopping triggered! Training halted due to no improvement.")
            #         # print(f"Best rmse: {self.best_rmse:.4f} at round {i - self.counter}")
            #         self.early_stop_flag = True
            #         # print(f"init round num{init_round} continue round num{i}")
            #         # if self.mode == 'baseline' or self.mode == 'dann':
            #         #     sys.exit("程序已终止")  # 终止训练循环
            #         # elif self.mode== 'dann+baseline':
            #
            # if self.early_stop_flag:
            #     break

            # if self.per_layer:
            #     for new_param, old_param in zip(self.model_backup.unique.parameters(),
            #                                     self.self.model.unique.parameters()):  # 可以选择
            #         old_param.data = new_param.data.clone()

            self.model.train()
            for i, (data1, data2) in enumerate(zip(self.source_trainloader, self.target_train_loader)):  # zip会在任一耗尽时停止
                print(f"client{self.id}  pretrain-local epoch: {epoch} ")
                # if i == 15:
                #     break
                x1, y1 = data1
                x2, y2 = data2
                x1, x2, y1, y2 = x1.to(self.device), x2.to(self.device), y1.to(self.device), y2.to(self.device)
                src_reg, src_pred, tgt_pred, src_feature, tgt_feature = self.model(x1, x2)

                src_reg_loss = self.prediction_loss(src_reg, y1)

                tgt_reg, _, _, _, _ = self.model(x2, x2)  # 应该几个模型？应该预期监控到tgt train loss也下降
                global_step = (self.global_round * max_local_epochs + epoch) * len(self.source_trainloader)#这里错了
                self.writer.add_scalar('pretrain-train/source_id' + str(self.source_id), torch.sqrt(src_reg_loss),
                                       global_step + i)  # 为什么baseline算法中只有source1的trainloss可复现
                # 明明在程序中从未明确指定源和目标。 首先排除dis_loss
                self.optimizer.zero_grad()
                src_reg_loss.backward()
                self.optimizer.step()

        self.counter = 0
        self.best_target_test_loss = 999
        #--------------------------------------预训练ES
        if self.mode == "Source-Only":
            sys.exit("Source-Only只有一阶段训练")
        self.model.encoder_frozen = copy.deepcopy(self.best_encoder_learning)
        # -------------------------------------------训练ET
        for epoch in range(
                max_local_epochs):  # 这里是finetune流程   和da的区别在于1.先train后test 2.我只关心target test metric 但也可以多观察些数据3.

            self.model.eval()
            for i, (data1, data2) in enumerate(zip(self.source_testloader, self.target_testloader)):
                x1, y1 = data1
                x2, y2 = data2
                x1, x2, y1, y2 = x1.to(self.device), x2.to(self.device), y1.to(self.device), y2.to(self.device)
                src_pred, _, _, _, _ = self.model(x1, x2)#看看ET+P预测的src准不准
                tgt_pred, f_target, f_source, loss_d, InfoNCE_loss = self.model(x2, x1)
                src_test_loss = self.prediction_loss(src_pred, y1)
                tgt_test_loss = self.prediction_loss(tgt_pred, y2)

                global_step_test = (self.global_round * max_local_epochs + epoch)
                self.writer.add_scalar(f'finetune-test/mmd{self.source_id}-{self.target_id}',
                                       mmd_rbf(f_target, f_source), global_step_test)
                self.writer.add_scalar('finetune-test/source_id' + str(self.source_id), torch.sqrt(src_test_loss),
                                       global_step_test)
                self.writer.add_scalar('finetune-test/target_id' + str(self.target_id), torch.sqrt(tgt_test_loss),
                                       global_step_test)
                self.writer.add_scalar('finetune-test/InfoNCE_loss', InfoNCE_loss, global_step_test)

            if torch.sqrt(tgt_test_loss) < self.best_target_test_loss:
                self.best_target_test_loss = torch.sqrt(tgt_test_loss)
                self.finetune_best_epoch = epoch
            #不清楚第二阶段多任务学习时test loss的动态变化，还是学满150吧。
            # if self.early_stop:
            #     if torch.sqrt(tgt_test_loss) < self.best_target_test_loss:
            #         self.best_target_test_loss = torch.sqrt(tgt_test_loss)
            #         self.counter = 0
            #     else:
            #         self.counter += 1
            #
            #     if self.counter >= self.patience:
            #         print("🔥🔥🔥 Early stopping triggered! Training halted due to no improvement.")
            #         # print(f"Best rmse: {self.best_rmse:.4f} at round {i - self.counter}")
            #         self.early_stop_flag = True
            #         # print(f"init round num{init_round} continue round num{i}")
            #         sys.exit("程序已终止")  # 终止训练循环

            self.model.train()
            for i, (data1, data2) in enumerate(zip(self.source_trainloader, self.target_train_loader)):  # zip会在任一耗尽时停止
                print(f"client{self.id}  finetune-local epoch: {epoch} ")
                # if i == 15:
                #     break
                x1, y1 = data1
                x2, y2 = data2
                x1, x2, y1, y2 = x1.to(self.device), x2.to(self.device), y1.to(self.device), y2.to(self.device)
                src_pred, _, _, _, _= self.model(x1, x2)
                tgt_pred, f_target, f_source, loss_d, InfoNCE_loss = self.model(x2, x1)
                src_reg_loss = self.prediction_loss(src_pred, y1)
                tgt_reg_loss = self.prediction_loss(tgt_pred, y2)
                global_step = (self.global_round * max_local_epochs + epoch) * len(self.target_train_loader)#错了
                self.writer.add_scalar('finetune-train/source_id' + str(self.source_id), torch.sqrt(src_reg_loss),
                                       global_step + i)  # 为什么baseline算法中只有source1的trainloss可复现
                self.writer.add_scalar('finetune-train/target_id' + str(self.target_id), torch.sqrt(tgt_reg_loss),
                                       global_step + i)  # 为什么baseline算法中只有source1的trainloss可复现
                # 明明在程序中从未明确指定源和目标。 首先排除dis_loss
                self.writer.add_scalar(f'finetune-train/mmd{self.source_id}-{self.target_id}',
                                       mmd_rbf(f_target, f_source), global_step)
                if self.mode == "CADA":
                    loss = loss_d + self.lambda_nce*InfoNCE_loss
                elif self.mode == "wo-InfoNCE":
                    loss = loss_d
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print(f"test loss={self.best_target_test_loss} at epoch{self.finetune_best_epoch}")


        # -------------------------------------------训练ET

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        source_train_data = read_client_data(self.dataset, self.source_id, self.args, is_train=True)
        target_train_data = read_client_data(self.dataset, self.target_id, self.args, is_train=True)
        return DataLoader(source_train_data, batch_size, drop_last=False, shuffle=True), DataLoader(target_train_data,
                                                                                                    batch_size,
                                                                                                    drop_last=False,
                                                                                                    shuffle=True)
        # return DataLoader(source_train_data, batch_size, drop_last=False, shuffle=False), DataLoader(target_train_data, batch_size, drop_last=False, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        source_test_data = read_client_data(self.dataset, self.source_id, self.args, is_train=False)
        target_test_data = read_client_data(self.dataset, self.target_id, self.args, is_train=False)
        return DataLoader(source_test_data, batch_size, drop_last=False, shuffle=True), DataLoader(target_test_data,
                                                                                                   batch_size,
                                                                                                   drop_last=False,
                                                                                                   shuffle=True)
        # return DataLoader(source_test_data, batch_size, drop_last=False, shuffle=False), DataLoader(target_test_data, batch_size, drop_last=False, shuffle=False)

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
