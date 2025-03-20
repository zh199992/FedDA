from system.client.clientFedAvg import clientAvg
from system.server.serverbase import Server


class serverAvg(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientAvg)
        print("Finished creating server and clients.")

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

            self.receive_models()

            self.aggregate_parameters()

        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()
