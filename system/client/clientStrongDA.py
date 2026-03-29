import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import read_client_data, read_client_data_centralized, MyDataset
from utils.metric import SF
import adamod
from system.client.clientbase import Client
import sys
from utils.mmdloss import mmd_rbf
import nni
import functools

def monitor_gpu_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)

        # 清空缓存，确保数据准确（可选，但推荐）
        torch.cuda.empty_cache()

        # 记录开始前状态
        before_alloc = torch.cuda.memory_allocated() / 1024 ** 2
        before_max = torch.cuda.max_memory_allocated() / 1024 ** 2

        print(f"\n🚀 [显存监控] 进入函数: {func.__name__}")
        print(f"   起始占用: {before_alloc:.2f} MB")

        result = func(*args, **kwargs)

        # 记录结束后状态
        after_alloc = torch.cuda.memory_allocated() / 1024 ** 2
        after_max = torch.cuda.max_memory_allocated() / 1024 ** 2

        print(f"✅ [显存监控] 离开函数: {func.__name__}")
        print(f"   最终占用: {after_alloc:.2f} MB (净增: {after_alloc - before_alloc:.2f} MB)")
        print(f"   过程峰值: {after_max:.2f} MB")
        print("-" * 30)

        return result

    return wrapper


class clientDANN(Client):
    def __init__(self, args, source_id, train_samples, test_samples, target_id, writer, **kwargs):
        self.source_id = source_id  # integer
        self.target_id = target_id
        super().__init__(args, source_id, train_samples, test_samples, writer, **kwargs)
        self.source_model = copy.deepcopy(self.model)
        self.target_model = copy.deepcopy(self.source_model)
        if hasattr(self.source_model.unique[0], 'flatten_parameters'):
            self.source_model.unique[0].flatten_parameters()
            self.target_model.unique[0].flatten_parameters()
        if self.args.optimizer_client == 'adamod':
            self.source_optimizer = adamod.AdaMod(self.source_model.parameters(), lr=self.learning_rate)
            self.target_optimizer = adamod.AdaMod(self.target_model.parameters(), lr=self.learning_rate)
        elif self.args.optimizer_client == 'adam':
            base_params = [
                {'params': self.source_model.F.parameters()},
                {'params': self.source_model.LHDR.parameters()},
                {'params': self.source_model.unique.parameters()}
            ]
            discriminator_params = [
                {
                    'params': self.source_model.discriminator.parameters(),
                    'lr': self.learning_rate * args.discriminator_lr  # 这里可以根据需要调整比例
                }
            ]
            self.source_optimizer = torch.optim.Adam(base_params + discriminator_params, lr=self.learning_rate)
            self.target_optimizer = torch.optim.Adam(self.target_model.parameters(), lr=self.learning_rate)
        elif self.args.optimizer_client == 'sgd':
            self.source_optimizer = torch.optim.SGD(self.source_model.parameters(), lr=self.learning_rate)
            self.target_optimizer = torch.optim.SGD(self.target_model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError
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
        self.gamma = args.gamma
        self.model_backup = copy.deepcopy(self.model)
        self.per_layer = None

    def train(self):
        self.target_model_pretraining()  # target model
        nni.report_intermediate_result(self.best_target_test_loss.item())
        self.create_tgt_feature_dataloader()
        self.source_model_pretraining()#source model
        # self.federated_averaging()
        # self.source_model_finetuning()
        for target_param, source_param in zip(self.target_model.F.parameters(), self.source_model.F.parameters()):
            target_param.data = source_param.data * self.miu_su + target_param.data * (1-self.miu_su)

        for target_param, source_param in zip(self.target_model.LHDR.parameters(), self.source_model.LHDR.parameters()):
            target_param.data = source_param.data * self.miu_su + target_param.data * (1-self.miu_su)

        for target_param, source_param in zip(self.target_model.unique.parameters(), self.source_model.unique.parameters()):
            target_param.data = source_param.data * self.miu_su + target_param.data * (1-self.miu_su)
        self.target_model_finetuning()
        nni.report_intermediate_result(self.best_target_test_loss.item())
        # nni.report_final_result(self.best_target_test_loss.item())

    @monitor_gpu_memory
    def federated_averaging(self):
        """
        对两个模型进行加权聚合
        alpha: model_a 的权重比例 (0 到 1 之间)
        """
        # 1. 创建一个新模型作为聚合后的容器
        # global_model = copy.deepcopy(model_a)

        # 2. 获取两个模型的参数字典
        state_dict_t = self.target_model.state_dict()
        state_dict_s = self.source_model.state_dict()

        # 3. 创建用于存放聚合参数的字典
        # combined_state_dict = global_model.state_dict()

        # 4. 遍历所有参数键值对进行加权聚合
        for key in state_dict_t:
            # 计算公式: W_global = alpha * W_a + (1 - alpha) * W_b
            state_dict_t[key] = self.miu_su * state_dict_s[key] + (1.0 - self.miu_su) * state_dict_t[key]

    @monitor_gpu_memory
    def create_tgt_feature_dataloader(self):
        self.target_model.eval()
        all_features = []
        with torch.no_grad():
            for i, data in enumerate(self.target_train_loader):
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                _, _, _, tgt_feature, _ = self.target_model(x, x, None, 'mode1')
                all_features.append(tgt_feature.cpu())

        final_features = torch.cat(all_features, dim=0)
        final_labels = torch.ones(len(final_features))##注意了 之前弄错了
        feature_dataset = MyDataset(final_features, final_labels)
        self.tgt_feature_loader = DataLoader(
            feature_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    def source_model_finetuning(self):
        pass

    @monitor_gpu_memory
    def source_model_pretraining(self):
        self.counter = 0
        self.best_target_test_loss = 999
        self.best_source_test_loss = 999
        self.source_model.train()

        max_local_epochs = self.local_epochs
        for epoch in range(max_local_epochs):
            self.source_model.eval()
            with torch.no_grad():
                for i, (data1, data2) in enumerate(zip(self.source_testloader, self.target_testloader)):
                    x1, y1 = data1
                    x2, y2 = data2
                    x1, x2, y1, y2 = x1.to(self.device), x2.to(self.device), y1.to(self.device), y2.to(self.device)
                    src_pred, _, _, src_feature, tgt_feature = self.source_model(x1, x2, None, 'mode1')
                    tgt_pred, _, _, _, _ = self.source_model(x2, x2, None, 'mode1')
                    src_test_loss = self.prediction_loss(src_pred, y1)
                    tgt_test_loss = self.prediction_loss(tgt_pred, y2)
                    global_step_test = (self.global_round * max_local_epochs + epoch)
                    self.writer.add_scalar(f'source_pretraining-test/mmd{self.source_id}-{self.target_id}',
                                           mmd_rbf(torch.flatten(src_feature, start_dim=1), torch.flatten(tgt_feature, start_dim=1)), global_step_test)
                    self.writer.add_scalar('source_pretraining-test/source_id' + str(self.source_id), torch.sqrt(src_test_loss),
                                           global_step_test)
                    self.writer.add_scalar('source_pretraining-test/target_id' + str(self.target_id), torch.sqrt(tgt_test_loss),
                                           global_step_test)


            if self.early_stop:
                # 假设 evaluate() 会把结果存到 self.rs_test_rmse 列表里
                # if src_test_loss < self.best_source_test_loss:
                #     self.best_source_test_loss = src_test_loss
                #     self.counter=0
                # else:
                #     self.counter += 1
                if torch.sqrt(src_test_loss) < self.best_source_test_loss:
                    self.best_source_test_loss = torch.sqrt(src_test_loss)
                    self.counter = 0
                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    print("🔥🔥🔥 Early stopping triggered! Training halted due to no improvement.")
                    # print(f"Best rmse: {self.best_rmse:.4f} at round {i - self.counter}")
                    self.early_stop_flag = True
                    # print(f"init round num{init_round} continue round num{i}")
                    # if self.mode == 'baseline' or self.mode == 'dann':
                    #     sys.exit("程序已终止")  # 终止训练循环
                    # elif self.mode== 'dann+baseline':
                    return
            # if self.early_stop_flag:
            #     break

            self.source_model.train()
            for i, (data1, data2, data3) in enumerate(zip(self.source_trainloader, self.target_train_loader,
                                                          self.tgt_feature_loader)):  # zip会在任一耗尽时停止
                print(f"client{self.source_id}  source_pretraining-local epoch: {epoch} batch{i} ")
                x1, y1 = data1
                x2, y2 = data2
                target_feature, target_label = data3
                x1, x2, y1, y2, target_feature, target_label = (
                    x1.to(self.device), x2.to(self.device), y1.to(self.device), y2.to(self.device),
                    target_feature.to(self.device), target_label.to(self.device))
                src_labels = torch.zeros(x1.size()[0]).type(torch.LongTensor).to(self.device)
                # tgt_labels = torch.ones(x2.size()[0]).type(torch.LongTensor).to(self.device)
                # x2 = torch.zeros_like(x2)#----------------------------
                src_reg, src_pred, tgt_pred, src_feature, tgt_feature = self.source_model(x1, None, target_feature.detach(), 'mode2')
                src_reg_loss = self.prediction_loss(src_reg, y1)
                # tgt_reg, _, _, _, _ = self.source_model(x2, x2, None, 'mode1')  # 应该几个模型？
                # tgt_reg_loss = self.prediction_loss(tgt_reg, y2)
                global_step = (self.global_round * max_local_epochs + epoch) * len(self.trainloader)
                self.writer.add_scalar('source_pretraining-train/source_id' + str(self.source_id), torch.sqrt(src_reg_loss),
                                       global_step + i)  # 为什么baseline算法中只有source1的trainloss可复现
                # 明明在程序中从未明确指定源和目标。 首先排除dis_loss
                dis_loss = self.adv_loss(src_pred, src_labels.long()) + self.adv_loss(tgt_pred, target_label.long())
                # mmd_loss = mmd_rbf(torch.flatten(src_feature, start_dim=1), torch.flatten(tgt_feature, start_dim=1))
                self.writer.add_scalar('source_pretraining-train/dis_loss' + f"{self.source_id}-{self.target_id}", dis_loss,
                                       global_step + i)
                self.writer.add_scalar(f'source_pretraining-train/mmd{self.source_id}-{self.target_id}',
                                       mmd_rbf(torch.flatten(src_feature, start_dim=1), torch.flatten(target_feature, start_dim=1)), global_step + i)
                self.source_optimizer.zero_grad()
                if self.mode == 'baseline':
                    loss = src_reg_loss
                elif self.mode == 'dann' or self.mode == "centralized":
                    loss = src_reg_loss + self.gamma* dis_loss
                    # loss = src_reg_loss + self.gamma * dis_loss
                elif self.mode == 'mmd':
                    loss = self.gamma * src_reg_loss + (1 - self.gamma) * mmd_loss
                    # loss = src_reg_loss + self.gamma * mmd_loss
                elif self.mode == 'dann+mmd':  # ???
                    # loss = self.gamma* src_reg_loss + (1-self.gamma) * dis_loss + (1-self.gamma) * mmd_loss
                    loss = src_reg_loss + self.gamma * dis_loss + self.gamma * mmd_loss
                elif self.mode == "mutual":
                    # loss = self.gamma* src_reg_loss + self.gamma* tgt_reg_loss + (1-self.gamma) * dis_loss
                    loss = src_reg_loss + tgt_reg_loss + (1 - self.gamma) / self.gamma * dis_loss
                    # loss = src_reg_loss + tgt_reg_loss + (1-self.gamma)/self.gamma * dis_loss
                else:
                    raise ValueError('Invalid mode')
                # elif self.mode == 'dann_adv':
                #     loss=src_reg_loss +
                # self.writer.add_scalar('train/source_id'+str(self.id),torch.sqrt(src_reg_loss),global_step+i)
                loss.backward()
                # if self.client_clip==True:
                #     grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
                self.source_optimizer.step()

    @monitor_gpu_memory
    def target_model_pretraining(self):
        self.counter = 0
        self.best_target_test_loss = 999
        self.best_source_test_loss = 999

        max_local_epochs = self.local_epochs
        print("max_epochs", max_local_epochs)

        # -----------------------------
        for epoch in range(max_local_epochs):

            self.target_model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.target_testloader):
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    tgt_pred, _, _, _, _ = self.target_model(x, x, None, 'mode1')
                    # tgt_pred, _, _ = self.target_model(x)
                    tgt_test_loss = self.prediction_loss(tgt_pred, y)
                    global_step_test = (self.global_round * max_local_epochs + epoch)

                    self.writer.add_scalar('target_pretraining-test/target_id' + str(self.target_id), torch.sqrt(tgt_test_loss),
                                           global_step_test)

            if self.early_stop:
                if torch.sqrt(tgt_test_loss) < self.best_target_test_loss:
                    self.best_target_test_loss = torch.sqrt(tgt_test_loss)
                    self.counter = 0
                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    print("🔥🔥🔥 Early stopping triggered! Training halted due to no improvement.")
                    # print(f"Best rmse: {self.best_rmse:.4f} at round {i - self.counter}")
                    self.early_stop_flag = True
                    # print(f"init round num{init_round} continue round num{i}")
                    # sys.exit("程序已终止")  # 终止训练循环
                    return
            self.target_model.train()
            for i, data in enumerate(self.target_train_loader):  # zip会在任一耗尽时停止
                print(f"client{self.target_id}  target_pretraining epoch: {epoch}  batch {i}")
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                tgt_reg, _, _, _, _ = self.target_model(x, x, None, 'mode1')
                # tgt_reg, _, _ = self.target_model(x)
                tgt_reg_loss = self.prediction_loss(tgt_reg, y)
                global_step = (self.global_round * max_local_epochs + epoch) * len(self.trainloader)
                self.writer.add_scalar('target_pretraining-train/target_id' + str(self.target_id), torch.sqrt(tgt_reg_loss),
                                       global_step + i)  # 为什么baseline算法中只有source1的trainloss可复现
                self.target_optimizer.zero_grad()
                loss = tgt_reg_loss
                loss.backward()
                self.target_optimizer.step()

    @monitor_gpu_memory
    def target_model_finetuning(self):
        self.counter = 0
        self.best_target_test_loss = 999
        self.best_source_test_loss = 999

        max_local_epochs = self.local_epochs
        print("max_epochs", max_local_epochs)
        # -----------------------------
        for epoch in range(max_local_epochs):

            self.target_model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.target_testloader):
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                    tgt_pred, _, _, _, _ = self.target_model(x, x, None, 'mode1')
                    tgt_test_loss = self.prediction_loss(tgt_pred, y)
                    global_step_test = (self.global_round * max_local_epochs + epoch)

                    self.writer.add_scalar('target_finetuning-test/target_id' + str(self.target_id), torch.sqrt(tgt_test_loss),
                                           global_step_test)

            if self.early_stop:
                if torch.sqrt(tgt_test_loss) < self.best_target_test_loss:
                    self.best_target_test_loss = torch.sqrt(tgt_test_loss)
                    self.counter = 0
                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    print("🔥🔥🔥 Early stopping triggered! Training halted due to no improvement.")
                    # print(f"Best rmse: {self.best_rmse:.4f} at round {i - self.counter}")
                    self.early_stop_flag = True
                    # print(f"init round num{init_round} continue round num{i}")
                    # sys.exit("程序已终止")  # 终止训练循环
                    return
            self.target_model.train()
            for i, data in enumerate(self.target_train_loader):  # zip会在任一耗尽时停止
                print(f"client{self.target_id}  target_finetuning epoch: {epoch} batch{i}")
                x, y = data
                x, y = x.to(self.device), y.to(self.device)
                tgt_reg, _, _, _, _ = self.target_model(x, x, None, 'mode1')
                tgt_reg_loss = self.prediction_loss(tgt_reg, y)
                global_step = (self.global_round * max_local_epochs + epoch) * len(self.trainloader)
                self.writer.add_scalar('target_finetuning-train/target_id' + str(self.target_id), torch.sqrt(tgt_reg_loss),
                                       global_step + i)  # 为什么baseline算法中只有source1的trainloss可复现
                self.target_optimizer.zero_grad()
                loss = tgt_reg_loss
                loss.backward()
                self.target_optimizer.step()

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        source_train_data = read_client_data(self.dataset, self.source_id, self.args, is_train=True)
        target_train_data = read_client_data(self.dataset, self.target_id, self.args, is_train=True, train_ratio=self.args.train_ratio)
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
