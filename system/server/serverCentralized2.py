from system.client.clientFedAvgiid import clientAvgiid############3为什么用这个，我对client类的要求是什么
from system.client.clientFedAvg import clientAvg
from utils.data_utils import read_client_data_centralized  #为什么要另外写
from utils.data_utils import calculate_split_sizes

from system.server.serverbase import Server
from utils.data_utils import visualize
import matplotlib.pyplot as plt
from utils.model_diff import calculate_l2_diff
import copy
import adamod
import torch
import torch.nn as nn
class serverCentralized(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)
        if self.args.optimizer_server == 'adamod':
            self.optimizer = adamod.AdaMod(self.global_model.parameters(), lr=self.lr)
        elif self.args.optimizer_client == 'adam':
            self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.lr)
        elif self.args.optimizer_client == 'sgd':
            self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
        self.trainset = read_client_data_centralized(self.dataset, self.args, is_train=True)
        self.testset = read_client_data_centralized(self.dataset, self.args, is_train=False)
        self.trainloader= torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size_server, shuffle=True)
        self.testloader= torch.utils.data.DataLoader(self.trainset, batch_size=1024, shuffle=True)
        self.loss = nn.MSELoss()
        # self.set_clients(clientAvgiid)
        self.set_clients(clientAvg)
        print("Finished creating server and clients.")

    def train(self):
        #1.云端训练 难点是把trainloader放到云端  2.测评模型 3.下发到client训练 4.测评
        for globalround in range(self.global_rounds+1):#+1是为了evaluate吗
            print(f"Round{globalround}")
            for epoch in range(self.args.server_epochs):
                print(f"global round {globalround} server epoch: {epoch} ")
                global_step=(globalround*self.args.server_epochs+epoch)*len(self.trainloader)
                for k, (x, y) in enumerate(self.trainloader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output, _, _ = self.global_model(x)
                    loss = self.loss(output, y)
                    self.writer.add_scalar('train/server_cent', torch.sqrt(loss), global_step + k)
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip:
                        grad = torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=100)
                    self.optimizer.step()

                for k, (x, y) in enumerate(self.testloader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output, _, _ = self.global_model(x)
                    loss = self.loss(output, y)
                    self.writer.add_scalar('test/server_cent', torch.sqrt(loss), globalround*self.args.server_epochs+epoch)#用来观察什么时候适合微调


            self.send_models()

            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            #     print("\nEvaluate global model")
            self.evaluate_before_finetune(round=globalround)

            for client in self.clients:
                client.global_round = globalround
                client.train()
            for client in self.clients:
                client.get_feature()
            # self.receive_models_features()
            self.evaluate(round=globalround)##为什么和train时最后一次testloss不一样？？？


        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()


    # def set_clients(self, clientObj):#设置了就有n个client作为server的属性     train_set怎么传给client对象的？ init时再调一遍read
    #     for i in range(1,1+self.num_clients):
    #         train_set = read_client_data_centralized(self.dataset, i,self.args, is_train=True)
    #         test_set = read_client_data_centralized(self.dataset, i, self.args,is_train=False)
    #         client = clientObj(self.args,
    #                            id=i, trainset=train_set, testset=test_set,
    #                            train_samples=len(train_set),
    #                            test_samples=len(test_set),writer=self.writer)
    #         self.clients.append(client)
    #
    #     return self.clients
    def evaluate_before_finetune(self, round):
        stats = self.test_metrics()
        total_test_loss = (torch.tensor([l*n for l,n in zip(stats[0],stats[1])])).sum()/(torch.tensor(stats[1]).sum())

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test loss: {:.4f}".format(torch.sqrt(total_test_loss)))
        for i in range(len(stats[0])):
            self.writer.add_scalar("test/round loss before finetune "+str(i+1), torch.sqrt(stats[0][i]), round)
        self.writer.add_scalar("test/average loss before finetune", torch.sqrt(total_test_loss), round)

        print("Test score:" ,stats[2])
        # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test loss: {:.4f}".format(np.std(accs)))
        # print("Std Test score: {:.4f}".format(np.std(aucs)))