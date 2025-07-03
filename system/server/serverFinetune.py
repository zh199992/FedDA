from system.client.clientFinetune import clientFinetune
from system.server.serverbase import Server
from utils.data_utils import visualize
import matplotlib.pyplot as plt
from utils.model_diff import calculate_l2_diff
import copy
import torch
import os

class serverFinetune(Server):
    # def __init__(self, device, server_config):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(clientFinetune)
        print("Finished creating server and clients.")

    def train(self):
        fig, axes = plt.subplots(7, 6)
        axes = axes.flatten()
        model_tmp=copy.deepcopy(self.global_model.LHDR)
        best_metric=[float('inf') for i in range(4)]
        data_dim = self.args.dp.split('-')[0]
        model_name=type(self.global_model).__name__+data_dim
        best_model_path=['/home/zhouheng/project/FedDA/models/weights/'+'local'+'/'+model_name+f"_{i+1}" for i in range(4)]
        os.makedirs(os.path.dirname('/home/zhouheng/project/FedDA/models/weights/'+'local'+'/'), exist_ok=True)

        self.global_model.load_state_dict(torch.load(best_model_path[3]))
        self.send_models()
        for i in range(self.global_rounds+1):#+1是为了evaluate吗
            print(f"Round{i}")
            # if i%self.eval_gap == 0:#控制输出训练效果的间隔   为什么是获取了才evaluate？
            #     print(f"\n-------------Round number: {i}-------------")
            #     print("\nEvaluate global model")
            self.evaluate(round=i)
            stats = self.test_metrics()
            for idx,client in enumerate(self.clients):
                if best_metric[idx]>torch.sqrt(stats[0][idx]):
                    best_metric[idx]=torch.sqrt(stats[0][idx])
                    # torch.save(client.model.state_dict(),best_model_path[idx])
                    print(f"round {i }client{idx+1}: 验证损失降低 ({best_metric[idx]:.4f}), 保存模型")

            # for client in self.clients:
            #     client.get_feature()
            # self.receive_models_features()
            # visualize(self.uploaded_middle_features,self.graph_path+"/Round{i}before training",axes[2*i])
            # axes[2*i].set_title(f"Plot {2*i + 1}")
            for client in self.clients:
                client.global_round=i
                client.train()
        print(best_metric)
            #     print(calculate_l2_diff(model_tmp,client.model.LHDR))
            #     client.get_feature()
            # self.receive_models_features()
            # visualize(self.uploaded_middle_features,self.graph_path+f"/Round{i}after training", axes[2*i+1] )
            # axes[2*i+1].set_title(f"Plot {2*i + 2}")

        # plt.tight_layout()
        # plt.show()
        # print("\nBest accuracy.")
        # print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()


