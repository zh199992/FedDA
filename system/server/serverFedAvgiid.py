from system.client.clientFedAvgiid import clientAvgiid
from system.server.serverbase import Server
from utils.data_utils import read_client_data_iid  #为什么要另外写
from utils.data_utils import calculate_split_sizes
#test data用整个和四分之一个有什么区别？
class serverAvgiid(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientAvgiid)
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):  # +1是为了evaluate吗
            print(f"Round{i}")
            self.send_models()
            # if i==0:
            #     self.evaluate(round=i)
            self.evaluate(round=i)

            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")


            for client in self.clients:
                client.global_round = i
                client.train()
            for client in self.clients:
                client.get_feature()
            self.receive_models_features()
            # self.evaluate(round=i)

            self.aggregate_parameters()

        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()
    def set_clients(self, clientObj):#设置了就有n个client作为server的属性     train_set怎么传给client对象的？ init时再调一遍read
        split_sizes_train, split_sizes_test=calculate_split_sizes(self.dataset, self.args, self.num_clients)
        tmp_train=0
        tmp_test=0
        for i in range(1,1+self.num_clients):
            train_set = read_client_data_iid(self.dataset, split_sizes_train, tmp_train, i,self.args, is_train=True)
            test_set = read_client_data_iid(self.dataset, split_sizes_test, tmp_test, i, self.args,is_train=False)
            client = clientObj(self.args,
                               id=i, trainset=train_set, testset=test_set,
                               train_samples=len(train_set),
                               test_samples=len(test_set),writer=self.writer)
            self.clients.append(client)
            tmp_train+=split_sizes_train[i-1]
            tmp_test+=split_sizes_test[i-1]
        return self.clients