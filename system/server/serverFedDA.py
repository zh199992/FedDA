from system.client.clientFedDA import clientDA
from system.server.serverbase import Server
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_utils import CombinedDataset, MyDataset
from utils.mmdloss import mmd_rbf

#改变1 不用receive_models改用receive_models_and_features 2 cloudda 3重写aggregate_parameters和add_parameters 和lr 4.云端的optimizer和dataset  global model也得改
#首先，我有四个client的特征，要减小他们四个被LHDR处理后的距离
class serverDA(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)
        self.global_model = copy.deepcopy(args.server_model)###########33
        self.set_clients(clientDA)
        print("Finished creating server and clients.")
        self.lr1=float(args.server_learning_rate.split(',')[0])
        self.lr2=float(args.server_learning_rate.split(',')[1])

        self.optimizer=torch.optim.Adam(self.global_model.parameters(), lr=self.lr1) #如果两个loss一起优化，只需要一个
        # self.optimizer_D=torch.optim.Adam(self.global_model.parameters(), lr=self.lr)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay
        self.advloss=nn.CrossEntropyLoss()
        self.lambda_mmd=1

    def train(self):
        for i in range(self.global_rounds+1):  # +1是为了evaluate吗
            print(f"Round{i}")
            self.send_models()

            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate(round=i)

            for client in self.clients:
                client.global_round = i
                client.train()

            self.receive_models_features()

            self.aggregate_parameters()

            self.cloud_da1(global_round=i)
        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()

    def cloud_da1(self,global_round):
        #1.得到标签 2.dataloader 3.训练 4.输出/画出结果
        for epoch in range(self.epoches):
            global_step = (global_round * (self.epoches) + epoch) * len(self.combined_loader)#如果batchsize128  1.4e6/1024=1.4e3个batch -se=10的话，云端有1.4e4个step
            for i, (batch_data, batch_domains) in enumerate(self.combined_loader):
                self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                batch_domains=batch_domains.to(self.device)
                domain_preds, feature_DI=self.global_model(batch_data)
                domain_labels = batch_domains
                adv_loss = self.advloss(domain_preds, domain_labels)
                mmd_loss = compute_mmd_loss(feature_DI, batch_domains)
                self.writer.add_scalar('train/server_mmd',mmd_loss,global_step+i)
                self.writer.add_scalar('train/server_adv',adv_loss,global_step+i)

                total_loss = adv_loss + self.lambda_mmd*mmd_loss
                total_loss.backward()
                self.optimizer.step()

    # def cloud_da2(self):  #一个其他采样方法


    def receive_models_features(self):
        # 断言语句 不满足会触发AssertionError
        assert (len(self.clients) > 0)

        active_clients = self.clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_feature_sets = [] #存四个dataset

        tot_samples = 0
        for i, client in enumerate(active_clients):
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
            uploaded_feature=torch.cat(client.shallow_feature,dim=0)
            dataset=MyDataset(uploaded_feature,uploaded_feature )
            self.uploaded_feature_sets.append(dataset)  ####改动

        self.combined_loader=DataLoader(CombinedDataset(self.uploaded_feature_sets), batch_size=self.batch_size, shuffle=True)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples #根据数据量


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model.F = copy.deepcopy(self.uploaded_models[0].F)
        for param in self.global_model.F.parameters():     #discriminator不能清零
            param.data.zero_()
        for param in self.global_model.LHDR.parameters():     #discriminator不能清零
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.F.parameters(), client_model.F.parameters()):
            server_param.data += client_param.data.clone() * w
        for server_param, client_param in zip(self.global_model.LHDR.parameters(), client_model.LHDR.parameters()):
            server_param.data += client_param.data.clone() * w


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
            mmd_loss += mmd_rbf(features_i, features_j)

    return mmd_loss