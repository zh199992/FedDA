from system.client.clientnewFedCADA import clientCADA
from system.server.serverbase import Server
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import CombinedDataset, MyDataset, visualize_features_with_rul, compute_rul_silhouette_score
from utils.mmdloss import mmd_rbf
import adamod
import nni

#改变1 不用receive_models改用receive_models_and_features 2 cloudda 3重写aggregate_parameters和add_parameters 和lr 4.云端的optimizer和dataset  global model也得改
#首先，我有四个client的特征，要减小他们四个被LHDR处理后的距离
#下发是通过global_model的，因此如果EDI不在global model里更新就是无效训练？
class servernewFedCADA(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)
        self.global_model = copy.deepcopy(args.server_model)###########33
        self.set_clients(clientCADA)
        print("Finished creating server and clients.")
        self.lr1=args.server_learning_rate
        # self.lr1=float(args.server_learning_rate.split(',')[0])
        # self.lr2=float(args.server_learning_rate.split(',')[1])
        if self.args.optimizer_server == 'adamod':
            self.optimizer = adamod.AdaMod(self.global_model.parameters(), lr=self.lr1)
        elif self.args.optimizer_server == 'adam':
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
        # self.advloss=nn.CrossEntropyLoss()#=nn.LogSoftmax()+nn.NLLLoss().#
        self.advloss = nn.BCEWithLogitsLoss()
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
        for i in range(self.global_rounds_init+1):#默认是0
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
            print(f"Round{i}")
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
                if self.args.new_da:
                    self.cloud_da2(global_round=i)
                else:
                    self.cloud_da1(global_round=i)

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
                domain_labels = batch_domains
                print(domain_preds.size(), domain_labels.size())
                adv_loss = self.advloss(domain_preds, domain_labels)
                mmd_loss = compute_mmd_loss(feature_DI, batch_domains)
                self.writer.add_scalar('train/server_mmd', mmd_loss, global_step + i)
                self.writer.add_scalar('train/server_adv', adv_loss, global_step + i)
                if self.DA_loss=='mmd':
                    total_loss = mmd_loss

                elif self.DA_loss=='adv':
                    total_loss = adv_loss

                elif self.DA_loss=='adv+mmd':
                    total_loss = adv_loss + self.lambda_mmd*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1
                    # total_loss = self.gamma*adv_loss/ + (1-self.gamma)*self.lambda_mmd*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1
                    total_loss = self.gamma*adv_loss + (1-self.gamma)*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1

                elif  self.DA_loss=='none':
                    total_loss = 0
                else:
                    raise NotImplementedError

                total_loss.backward()
                self.optimizer.step()
            #-------------------------------------------------------
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

    def cloud_da2(self,global_round):#输入shallow，输出middle和固定的middle算loss。middle是随机抽还是固定一个batch？
        #1.得到标签 2.dataloader 3.训练 4.输出/画出结果
        for clientid in range(self.num_clients):
            for epoch in range(self.epoches):
                global_step = (global_round * self.epoches + epoch) * len(self.cloud_shallow_loader_list[clientid])#1.4e6/1024=1.4e3个batch -se=10的话，云端有1.4e4个step 143533/1024=141
                for i, ((batch_data, _), (features_middle,_)) in enumerate(zip(self.cloud_shallow_loader_list[clientid],
                                                               self.combined_source_loader_list[clientid] )):
                    self.optimizer.zero_grad()
                    batch_data = batch_data.to(self.device)
                    # batch_domains=batch_domains.to(self.device)
                    domain_preds, feature_DI=self.global_model(batch_data, features_middle, clientid)#这里也是改动
                    # domain_labels = batch_domains
                    # print(domain_preds.size(), domain_labels.size())
                    domain_labels = torch.cat([torch.ones([len(batch_data),1]), torch.zeros([len(features_middle),1])]).to(self.device)
                    adv_loss = self.advloss(domain_preds, domain_labels)
                    mmd_loss = mmd_rbf(feature_DI, features_middle)
                    #-----------------------------------------------------------------------
                    self.writer.add_scalar(f'train/client{clientid+1}server_mmd', mmd_loss, global_step + i)
                    self.writer.add_scalar(f'train/client{clientid+1}server_adv', adv_loss, global_step + i)
                    if self.DA_loss=='mmd':
                        total_loss = mmd_loss
                    elif self.DA_loss=='adv':
                        total_loss = adv_loss
                    elif self.DA_loss=='adv+mmd':
                        total_loss = adv_loss + self.lambda_mmd*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1
                        # total_loss = self.gamma*adv_loss + (1-self.gamma)*mmd_loss#gamma    (1-gamma     ) [0,1] 0.1

                    elif  self.DA_loss=='none':
                        total_loss = 0
                    else:
                        raise NotImplementedError

                    total_loss.backward()
                    self.optimizer.step()
                #-------------------------------------------------------
                if self.learning_rate_decay:
                    self.learning_rate_scheduler.step()


    def receive_models_features(self):
        # 断言语句 不满足会触发AssertionError
        assert (len(self.clients) > 0)

        active_clients = self.clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_shallow_sets = [] #存四个dataset
        self.uploaded_middle_features = []

        self.cloud_shallow_loader_list= []


        tot_samples = 0
        for i, client in enumerate(active_clients):
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)#整个上传
            uploaded_shallow=torch.cat(client.shallow_feature,dim=0)
            dataset=MyDataset(uploaded_shallow,uploaded_shallow)
            self.uploaded_shallow_sets.append(dataset)  ####改动

            self.cloud_shallow_loader_list.append(DataLoader(dataset, batch_size=self.batch_size, shuffle=True))

            uploaded_middle=torch.cat(client.middle_feature,dim=0)
            dataset = MyDataset(uploaded_middle, uploaded_middle)
            self.uploaded_middle_features.append(dataset)

        self.combined_source_loader_list = []
        self.combined_loader=DataLoader(CombinedDataset(
            self.uploaded_shallow_sets), batch_size=self.batch_size, shuffle=True)
        for i in range(self.num_clients):
            self.combined_source_loader_list.append(
                DataLoader(CombinedDataset(
                    self.uploaded_middle_features[:i]
                    + self.uploaded_middle_features[i + 1:]), batch_size=self.batch_size, shuffle=True)
            )

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples #根据数据量


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        # self.global_model.F = copy.deepcopy(self.uploaded_models[0].F)
        if self.args.F_FedAvg:
            for param in self.global_model.F.parameters():     #discriminator不能清零
                param.data.zero_()


        for idx, (w, client_model) in enumerate(zip(self.uploaded_weights, self.uploaded_models)):
            self.add_parameters(w, client_model)#如果要把四个client的E放到一个server model的e_list要在哪里做
            for new_param, old_param in zip(client_model.LHDR.parameters(), self.global_model.LHDR_list[idx].parameters()):#可以选择
                old_param.data = new_param.data.clone()

    def add_parameters(self, w, client_model):
        if self.args.F_FedAvg:
            for server_param, client_param in zip(self.global_model.F.parameters(), client_model.F.parameters()):
                server_param.data += client_param.data.clone() * w


    def soft_update(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.soft_update(self.global_model)


def compute_mmd_loss(features, domain_labels, sigma=1.0):#似乎不受不同域的样本数量影响
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

