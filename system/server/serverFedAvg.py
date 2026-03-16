from system.client.clientFedAvg import clientAvg
from system.server.serverbase import Server
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_diff import calculate_l2_diff, aggregate_with_sawa
import nni

class serverAvg(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientAvg)
        print("Finished creating server and clients.")

    def train(self):
        # self.RUL_CE()
        # sys.exit("完成")
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
            nni.report_intermediate_result(self.test_avg_loss)
            if self.args.client_lr_decay:
                for client in self.clients:
                    client.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=client.optimizer,
                        gamma=client.args.learning_rate_decay_gamma
                    )
                    for param_group in client.optimizer.param_groups:
                        param_group['lr'] = client.args.local_learning_rate

            for client in self.clients:
                client.global_round = i
                client.train()
            for client in self.clients:
                client.get_feature()
            self.receive_models_features()
            # self.uploaded_weights=[0.25,  0.25, 0.25, 0.25]##改动
            if self.args.algorithm == 'SAWA':
                self.uploaded_lhdrs = [nn.Sequential(*(list(i.F) + list(i.LHDR))) for i in self.uploaded_models]
                self.uploaded_weights = aggregate_with_sawa(self.uploaded_lhdrs)#
                for n in range(len(self.uploaded_weights)):
                    self.writer.add_scalar(f'train/agg_weight{n+1}', self.uploaded_weights[n], i)

            print(self.uploaded_weights)
            self.aggregate_parameters()

            if not self.args.fedeval:
                self.evaluate(round=i+1)

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
                    break  # 终止训练循环

        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()

