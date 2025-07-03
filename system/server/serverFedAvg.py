from system.client.clientFedAvg import clientAvg
from system.server.serverbase import Server
import torch
from utils.model_diff import calculate_l2_diff


class serverAvg(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientAvg)
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):  # +1是为了evaluate吗
            print(f"Round{i}")
            print('before download',calculate_l2_diff(self.clients[0].model, self.global_model))
            self.send_models()
            print('after download', calculate_l2_diff(self.clients[0].model, self.global_model))

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
            # self.uploaded_weights=[0.25,  0.25, 0.25, 0.25]##改动
            self.aggregate_parameters()

            if not self.args.fedeval:
                self.evaluate(round=i+1)

        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()
