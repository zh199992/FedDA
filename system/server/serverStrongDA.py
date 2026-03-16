from system.client.clientStrongDA import clientDANN
#---------------------------
from system.server.serverbase import Server
from utils.data_utils import read_client_data
import torch
#1.4个client，但是我通过超参数指定两个来训练   2.其实可以fedeval和indeval同步  只要画在两张图里就行 3.train代码的改变写在哪里
class serverStrongDA(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)
        self.source_id=args.source_id
        self.target_id=args.target_id
        self.set_clients(clientDANN)#一个client
        print("Finished creating server and clients.")


    def train(self):
        for i in range(0, 1):  # +1是为了evaluate吗
            print(f"Round{i}")
            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            # self.evaluate(round=i)


            for client in self.clients:
                client.global_round = i
                client.train()
            print('??')
            # for client in self.clients:
            #     client.get_feature()
        # self.evaluate(round=i+1)

    def set_clients(self, clientObj):#重写，我需要指定source和target
        source_train_set = read_client_data(self.dataset, self.source_id,self.args, is_train=True)
        target_train_set = read_client_data(self.dataset, self.target_id,self.args, is_train=True)
        source_test_set = read_client_data(self.dataset, self.source_id, self.args,is_train=False)
        target_test_set = read_client_data(self.dataset, self.target_id, self.args,is_train=False)
        client = clientObj(self.args,
                           source_id=self.source_id,
                           train_samples=len(source_train_set),
                           test_samples=len(source_test_set), target_id = self.target_id, writer=self.writer)
        self.clients.append(client)

        return self.clients