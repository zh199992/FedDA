from system.client.clientFedAvg import clientAvg
from system.server.serverbase import Server
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import CombinedDataset, MyDataset, visualize_features_with_rul, compute_rul_silhouette_score, regresssion_feature, plot_all_clients_features
from utils.mmdloss import mmd_rbf
import adamod
import nni

#改变1 不用receive_models改用receive_models_and_features 2 cloudda 3重写aggregate_parameters和add_parameters 和lr 4.云端的optimizer和dataset  global model也得改
#首先，我有四个client的特征，要减小他们四个被LHDR处理后的距离
#下发是通过global_model的，因此如果EDI不在global model里更新就是无效训练？
class serverDA(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)
        self.global_model = copy.deepcopy(args.server_model)###########33
        self.set_clients(clientAvg)
        print("Finished creating server and clients.")
        self.lr1=args.server_learning_rate
        # self.lr1=float(args.server_learning_rate.split(',')[0])
        # self.lr2=float(args.server_learning_rate.split(',')[1])
        if self.args.optimizer_server == 'adamod':
            self.optimizer = adamod.AdaMod(self.global_model.parameters(), lr=self.lr1)
        elif self.args.optimizer_server == 'adam':
            base_params = [
                {'params': self.global_model.LHDR.parameters()},
            ]
            discriminator_params = [
                {
                    'params': self.global_model.discriminator.parameters(),
                    'lr': self.lr1 * args.discriminator_lr  # 这里可以根据需要调整比例
                }
            ]
            self.source_optimizer = torch.optim.Adam(base_params + discriminator_params, lr=self.lr1)
            self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.lr1)
        elif self.args.optimizer_server == 'sgd':
            self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.lr1)
        else:
            raise NotImplementedError
        # self.optimizer_D=torch.optim.Adam(self.global_model.parameters(), lr=self.lr)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.server_lr_decay
        self.advloss=nn.CrossEntropyLoss()#=nn.LogSoftmax()+nn.NLLLoss().
        self.lambda_mmd=args.lambda_mmd
        self.gamma=args.gamma
        self.DA_loss=args.DA_loss
        self.global_rounds_init=args.global_rounds_init

        # ---------------------- Early Stopping Variables ----------------------
        self.early_stop = getattr(args, 'early_stop', False)         # 是否启用早停
        self.pretrain_early_stop = getattr(args, 'pretrain_early_stop', False)
        self.patience = 5              # 容忍多少轮不提升
        self.counter = 0                                             # 计数器
        self.early_stop_flag = False                                 # 是否触发早停
        # ----------------------------------------------------------------------

    def train(self):
        init_round = self.global_rounds_init
        for i in range(self.global_rounds_init+1):
            print(f"Pretraining Round{i}")
            print("\nEvaluate global model")
            if self.args.fedeval:
                self.evaluate(round=i)
            else:
                if i==0:
                    self.evaluate(round=i)
            if self.args.client_lr_decay:
                for client in self.clients:
                    client.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=client.optimizer,
                        gamma=client.args.learning_rate_decay_gamma
                    )

            for client in self.clients:
                client.global_round = i
                client.train()
            for client in self.clients:
                client.get_feature()
            self.receive_models_features()
            if not self.args.fedeval:
                self.evaluate(round=i+1)

            if self.pretrain_early_stop:
                # 假设 evaluate() 会把结果存到 self.rs_test_rmse 列表里
                current_rmse = self.rs_test_rmse[-1] if len(self.rs_test_rmse) > 0 else 999
                print(current_rmse,self.best_rmse)
                if current_rmse < self.best_rmse-0.1:
                    self.best_rmse = current_rmse
                    self.counter = 0  # 重置计数器
                    # 可选：保存当前最优模型
                    # self.save_global_model(model_name="best_global_model.pth")
                else:
                    self.counter += 1
                    print(f"[Early Stop] No improvement in accuracy. "
                          f"Counter: {self.counter}/{self.patience}")

                if self.counter >= self.patience:
                    print("🔥🔥🔥 Early stopping triggered! Training halted due to no improvement.")
                    print(f"Best rmse: {self.best_rmse:.4f} at round {i - self.counter}")
                    self.early_stop_flag = True
                    init_round=i
                    break  # 终止训练循环

#---------------------------------------------------------------------
        self.aggregate_parameters()
#----------------------------------------------------------------------
        self.counter=0
        for i in range(init_round+1,self.global_rounds+1):  # +1是为了evaluate吗
            print(f"Training Round{i}")
            if self.args.soft_update:
                self.soft_update()
            else:
                self.send_models()
            print("\nEvaluate global model")
            if self.args.fedeval:
                self.evaluate(round=i)
            else:
                if i==0:
                    self.evaluate(round=i)

            if self.args.client_lr_decay:
                for client in self.clients:
                    client.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=client.optimizer,
                        gamma=client.args.learning_rate_decay_gamma
                    )

            for client in self.clients:
                client.global_round = i
                client.train()
            for client in self.clients:
                client.get_feature()
            self.receive_models_features()

            self.aggregate_parameters()

            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=self.args.learning_rate_decay_gamma)
            if self.args.enable_cloud_da:
                print(f"DEBUG: Executing cloud_da1 at round {i}")
                self.cloud_da1(global_round=i)
            else:
                print("DEBUG: enable_cloud_da is FALSE, skipping...")
            if not self.args.fedeval:
                self.evaluate(round=i+1)

            # ========================================================================================================
            # === 🛑🛑🛑 EARLY STOPPING CHECKPOINT (MONITOR TEST ACCURACY) 🛑🛑🛑 ===================================
            # ========================================================================================================
            if self.early_stop:   
                # 假设 evaluate() 会把结果存到 self.rs_test_rmse 列表里
                current_rmse = self.rs_test_rmse[-1] if len(self.rs_test_rmse) > 0 else 999
                print(current_rmse,self.best_rmse)
                if current_rmse < self.best_rmse-0.1:
                    self.best_rmse = current_rmse
                    self.counter = 0  # 重置计数器
                    # 可选：保存当前最优模型
                    # self.save_global_model(model_name="best_global_model.pth")
                else:
                    self.counter += 1
                    print(f"[Early Stop] No improvement in accuracy. "
                          f"Counter: {self.counter}/{self.patience}")

                if self.counter >= self.patience:
                    print("🔥🔥🔥 Early stopping triggered! Training halted due to no improvement.")
                    print(f"Best rmse: {self.best_rmse:.4f} at round {i - self.counter}")
                    self.early_stop_flag = True
                    print(f"init round num{init_round} continue round num{i}")
                    break  # 终止训练循环
            # ========================================================================================================
            # === ✅ END OF EARLY STOPPING LOGIC =====================================================================
            # ========================================================================================================

        # print(plot_all_clients_features(self.uploaded_middle_features, self.uploaded_labels))
        scores = plot_all_clients_features(self.uploaded_middle_features, self.uploaded_labels)
        print(scores)
        # print("\nBest accuracy.")
        # print(max(self.rs_test_rmse))

        # self.save_results()
        # self.save_global_model()

    def cloud_da1(self,global_round):
        #1.得到标签 2.dataloader 3.训练 4.输出/画出结果
        for epoch in range(self.epoches):
            global_step = (global_round * self.epoches + epoch) * len(self.combined_loader)#1.4e6/1024=1.4e3个batch -se=10的话，云端有1.4e4个step 143533/1024=141
            for i, (batch_data, batch_domains) in enumerate(self.combined_loader):
                self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                batch_domains=batch_domains.to(self.device)
                domain_preds, feature_DI=self.global_model(batch_data)
                print(batch_domains,domain_preds,feature_DI)
                domain_labels = batch_domains
                adv_loss = self.advloss(domain_preds, domain_labels)
                mmd_loss = compute_mmd_loss(feature_DI, batch_domains)
                self.writer.add_scalar('train/server_mmd', mmd_loss, global_step + i)
                self.writer.add_scalar('train/server_adv', adv_loss, global_step + i)
                if self.DA_loss=='mmd':
                    total_loss = mmd_loss
                    # adv_loss = self.advloss(domain_preds, domain_labels)
                    # mmd_loss = compute_mmd_loss(feature_DI, batch_domains)
                    # self.writer.add_scalar('train/server_mmd', mmd_loss, global_step + i)
                    # self.writer.add_scalar('train/server_adv', adv_loss, global_step + i)
                    # mmd_loss=mmd_loss/mmd_loss.detach()
                    # mmd_loss.backward()
                    # self.optimizer.step()
                elif self.DA_loss=='adv':
                    total_loss = adv_loss
                    # adv_loss = self.advloss(domain_preds, domain_labels)
                    # mmd_loss = compute_mmd_loss(feature_DI, batch_domains)
                    # self.writer.add_scalar('train/server_mmd', mmd_loss, global_step + i)
                    # self.writer.add_scalar('train/server_adv', adv_loss, global_step + i)
                    # adv_loss=adv_loss/adv_loss.detach()
                    # adv_loss.backward()
                    # self.optimizer.step()
                elif self.DA_loss=='adv+mmd':
                    total_loss = adv_loss + self.lambda_mmd*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1
                    # total_loss = self.gamma*adv_loss/ + (1-self.gamma)*self.lambda_mmd*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1
                    total_loss = self.gamma*adv_loss + (1-self.gamma)*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1
                    # adv_loss = self.advloss(domain_preds, domain_labels)
                    # adv_loss_normalized = self.gamma*adv_loss/adv_loss.detach()
                    # adv_loss_normalized.backward()
                    # for name, param in self.global_model.named_parameters():
                    #     if param.grad is not None:
                    #         self.writer.add_histogram(f'{name}/adv_loss', param.grad, 0)
                    # self.optimizer.step()
                    # self.optimizer.zero_grad()
                    # #--------------这样应该可以创建新的计算图  实际应用再改会一起优化就行
                    # batch_data = batch_data.to(self.device)
                    # batch_domains = batch_domains.to(self.device)
                    # domain_preds, feature_DI = self.global_model(batch_data)
                    # mmd_loss = compute_mmd_loss(feature_DI, batch_domains)
                    # self.writer.add_scalar('train/server_mmd', mmd_loss, global_step + i)
                    # self.writer.add_scalar('train/server_adv', adv_loss, global_step + i)
                    # mmd_loss_normalized = (1-self.gamma)*mmd_loss/mmd_loss.detach()#gamma    (1-gamma     ) [0,1] 0.1
                    # mmd_loss_normalized.backward()
                    # for name, param in self.global_model.named_parameters():
                    #     if param.grad is not None:
                    #         self.writer.add_histogram(f'{name}/mmd_loss', param.grad, 0)
                    # self.optimizer.step()
                    # self.optimizer.zero_grad()
                elif  self.DA_loss=='none':
                    total_loss = 0
                else:
                    raise NotImplementedError

                total_loss.backward()
                self.optimizer.step()
            #-------------------------------------------------------
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

    # def cloud_da2(self):  #一个其他采样方法


    def receive_models_features(self):
        # 断言语句 不满足会触发AssertionError
        assert (len(self.clients) > 0)

        active_clients = self.clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_shallow_sets = [] #存四个dataset
        self.uploaded_middle_features = []
        self.uploaded_labels = []

        tot_samples = 0
        for i, client in enumerate(active_clients):
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)#整个上传
            uploaded_shallow=torch.cat(client.shallow_feature,dim=0)
            dataset=MyDataset(uploaded_shallow,uploaded_shallow)
            self.uploaded_shallow_sets.append(dataset)  ####改动

            uploaded_middle=torch.cat(client.middle_feature,dim=0)
            self.uploaded_middle_features.append(uploaded_middle)
            uploaded_label=torch.cat(client.label,dim=0)
            self.uploaded_labels.append(uploaded_label)

        self.combined_loader=DataLoader(CombinedDataset(self.uploaded_shallow_sets), batch_size=self.batch_size, shuffle=True)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples #根据数据量


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # self.global_model.F = copy.deepcopy(self.uploaded_models[0].F)
        if self.args.F_FedAvg:
            for param in self.global_model.F.parameters():     #discriminator不能清零
                param.data.zero_()
        if self.args.EDI_FedAvg:
            for param in self.global_model.LHDR.parameters():     #discriminator不能清零
                param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        if self.args.F_FedAvg:
            for server_param, client_param in zip(self.global_model.F.parameters(), client_model.F.parameters()):
                server_param.data += client_param.data.clone() * w
        if self.args.EDI_FedAvg:
            for server_param, client_param in zip(self.global_model.LHDR.parameters(), client_model.LHDR.parameters()):
                server_param.data += client_param.data.clone() * w

    def soft_update(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.soft_update(self.global_model)


def compute_mmd_loss(features, domain_labels, sigma=1.0):
    domains = torch.unique(domain_labels)#所有元素放到一个tensor
    mmd_loss = 0.0

    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            domain_i = domains[i].item()
            domain_j = domains[j].item()

            # 提取属于 domain_i 和 domain_j 的特征
            features_i = features[domain_labels == domain_i]#得到bool tensor
            features_j = features[domain_labels == domain_j]

            # 计算 MMD
            mmd_loss = mmd_loss+mmd_rbf(features_i, features_j)

    return mmd_loss

