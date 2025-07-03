from system.client.clientDANN import clientDANN
from system.server.serverbase import Server
import torch
#1.4个client，但是我通过超参数指定两个来训练   2.其实可以fedeval和indeval同步  只要画在两张图里就行 3.train代码的改变写在哪里
class serverDANN(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientDANN)
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):  # +1是为了evaluate吗
            print(f"Round{i}")
            self.send_models()

            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            if self.args.fedeval:
                self.evaluate(round=i)
            else:
                if i==0:
                    self.evaluate(round=i)

            # for client in self.clients:
            #     client.optimizer = torch.optim.Adam(client.model.parameters(), lr=client.learning_rate)  # 如果是别的方案就不能提前设置
            #     client.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            #         optimizer=client.optimizer,
            #         gamma=client.args.learning_rate_decay_gamma
            #     )
            for client in self.clients:
                client.global_round = i
                client.train()
            for client in self.clients:
                client.get_feature()
            self.receive_models_features()

            self.aggregate_parameters()

            if not self.args.fedeval:
                self.evaluate(round=i+1)

        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()
