from system.client.clientGHDR import clientGHDR
from system.server.serverbase import Server
import torch

class serverGHDR(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientGHDR)
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):  # +1是为了evaluate吗
            print(f"Round{i}")
            self.send_models()#要改

            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            if self.args.fedeval:
                self.evaluate(round=i)
            else:
                if i==0:
                    self.evaluate(round=i)

            if self.args.client_lr_decay:
                for client in self.clients:
                    client.learning_rate_scheduler1 = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=client.optimizer1,
                        gamma=client.args.learning_rate_decay_gamma
                    )
                    for param_group in client.optimizer1.param_groups:
                        param_group['lr'] = client.args.local_learning_rate
                    client.learning_rate_scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=client.optimizer2,
                        gamma=client.args.learning_rate_decay_gamma
                    )
                    for param_group in client.optimizer2.param_groups:
                        param_group['lr'] = client.args.local_learning_rate

            for client in self.clients:
                client.global_round = i
                client.train()#两个阶段做在这里
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
