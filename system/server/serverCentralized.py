from system.client.clientFedAvgiid import clientAvgiid############3为什么用这个，我对client类的要求是什么
from utils.data_utils import read_client_data_centralized  #为什么要另外写
from utils.data_utils import calculate_split_sizes

from system.server.serverbase import Server
from utils.data_utils import visualize
import matplotlib.pyplot as plt
from utils.model_diff import calculate_l2_diff
import copy

class serverCentralized(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)
        self.num_clients=1
        self.set_clients(clientAvgiid)
        print("Finished creating server and clients.")

    def train(self):
        self.send_models()
        for i in range(self.global_rounds+1):#+1是为了evaluate吗
            print(f"Round{i}")
            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            #     print("\nEvaluate global model")
            if i==0:
                self.evaluate(round=i)

            for client in self.clients:
                client.global_round = i
                client.train()
            for client in self.clients:
                client.get_feature()
            self.receive_models_features()
            self.evaluate(round=i+1)


        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()


    def set_clients(self, clientObj):#设置了就有n个client作为server的属性     train_set怎么传给client对象的？ init时再调一遍read
        for i in range(1,1+self.num_clients):
            train_set = read_client_data_centralized(self.dataset, i,self.args, is_train=True)
            test_set = read_client_data_centralized(self.dataset, i, self.args,is_train=False)
            client = clientObj(self.args,
                               id=i, trainset=train_set, testset=test_set,
                               train_samples=len(train_set),
                               test_samples=len(test_set),writer=self.writer)
            self.clients.append(client)

        return self.clients