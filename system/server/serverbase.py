import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data  #为什么要另外写
import torch.utils.tensorboard as tb

import nni
from utils.root import find_project_root
import numpy as np
import torch
import matplotlib.pyplot as plt
import functools
import inspect
import os


def monitor_gpu_memory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)

        torch.cuda.empty_cache()

        before_alloc = torch.cuda.memory_allocated() / 1024 ** 2
        before_max = torch.cuda.max_memory_allocated() / 1024 ** 2

        # 获取函数所属的文件路径和类名（如果是方法）
        func_file = inspect.getfile(func)
        func_filename = os.path.basename(func_file)

        # 判断是否是类方法，如果是则获取类名
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            func_location = f"{func_filename}::{class_name}.{func.__name__}"
        else:
            func_location = f"{func_filename}::{func.__name__}"

        print(f"\n🚀 [显存监控] 进入函数: {func_location}")
        print(f"   起始占用: {before_alloc:.2f} MB")

        result = func(*args, **kwargs)

        after_alloc = torch.cuda.memory_allocated() / 1024 ** 2
        after_max = torch.cuda.max_memory_allocated() / 1024 ** 2

        print(f"✅ [显存监控] 离开函数: {func_location}")
        print(f"   最终占用: {after_alloc:.2f} MB (净增: {after_alloc - before_alloc:.2f} MB)")
        print(f"   过程峰值: {after_max:.2f} MB")
        print("-" * 30)

        return result

    return wrapper

class Server(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_model = copy.deepcopy(args.model)###########33
        self.dp=args.dp
        self.batch_size=args.batch_size_server
        self.global_rounds=args.global_rounds
        self.epoches=int(args.server_epochs)###############################
        # self.lr=float(args.server_learning_rate.split(',')[0])
        self.lr=args.server_learning_rate
        self.schedule=args.server_schedule
        self.clip=args.server_clip
        self.algorithm = args.algorithm
        self.num_clients=args.num_clients
        self.dp=args.dp

        self.clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_loss = []
        self.rs_test_score = []
        self.rs_train_loss = []

        # graph_path = '/home/zhouheng/project/FedDA/logs/test1/' + self.algorithm + '/' + TIMESTAMP + self.args.model_name  # localupdateEDI
        # self.graph_path='/home/zhouheng/project/FedDA/logs/test1/' + self.algorithm + '/'+args.aim+'/'+args.TIMESTAMP
        root_dir = find_project_root('FedDA')
        #第一种路径  先分config和graph，再接试验任务。 第二种路径，先按任务分类，再config和graph
        # directory = 'test2/config/'  # 比画图多一个/config/
        if getattr(args, 'sub_algorithm', None) is None:
            self.graph_path = os.path.join(root_dir, "logs", 'graph',  # 这里示意一下，实际在serverbase-init 生效
                                          args.aim, nni.get_experiment_id(), args.algorithm, nni.get_trial_id() + '-' + args.TIMESTAMP)
        else:
            self.graph_path = os.path.join(root_dir, "logs", 'graph',  # 这里示意一下，实际在serverbase-init 生效
                                          args.aim, nni.get_experiment_id(), args.sub_algorithm, nni.get_trial_id() + '-' + args.TIMESTAMP)
        # self.graph_path=root_dir+'/logs/0730/graph/' + self.algorithm + '/'+args.aim+'/'+args.TIMESTAMP
        self.writer = tb.SummaryWriter(self.graph_path)  # tensorboard文件存储的文件夹

        self.test_avg_loss=999
        self.rs_test_rmse=[]
        self.best_rmse=999
        self.client_best_loss=[999,999,999,999]
        # ---------------------- Early Stopping Variables ----------------------
        self.early_stop = getattr(args, 'early_stop', False)         # 是否启用早停
        self.pretrain_early_stop = getattr(args, 'pretrain_early_stop', False)
        self.patience = 5              # 容忍多少轮不提升
        self.counter = 0                                             # 计数器
        self.early_stop_flag = False                                 # 是否触发早停
        # ----------------------------------------------------------------------

    @monitor_gpu_memory
    def set_clients(self, clientObj):#设置了就有n个client作为server的属性
        for i in range(1,1+self.num_clients):
            train_set = read_client_data(self.dataset, i,self.args, is_train=True)
            test_set = read_client_data(self.dataset, i, self.args,is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_set),
                               test_samples=len(test_set),writer=self.writer)
            self.clients.append(client)

        return self.clients

    @monitor_gpu_memory
    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)

    @monitor_gpu_memory
    def receive_models_features(self):
        # 断言语句 不满足会触发AssertionError
        assert (len(self.clients) > 0)

        active_clients = self.clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_middle_features = []

        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

            # uploaded_middle=torch.cat(client.middle_feature,dim=0)
            uploaded_middle = torch.cat([f.detach() for f in client.middle_feature], dim=0)
            self.uploaded_middle_features.append(uploaded_middle)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    @monitor_gpu_memory
    def evaluate(self, round):
        stats = self.test_metrics()
        # total_test_loss = (torch.tensor([l*n for l,n in zip(stats[0],stats[1])])).sum()/(torch.tensor(stats[1]).sum())
        #
        # # print("Averaged Train Loss: {:.4f}".format(train_loss))
        # print("Averaged Test loss: {:.4f}".format(torch.sqrt(total_test_loss)))
        for i in range(len(stats[0])):
            self.writer.add_scalar("test/round loss "+str(i+1), torch.sqrt(stats[0][i]), round)
            if torch.sqrt(stats[0][i])<self.client_best_loss[i]:#stats里是MSE
                self.client_best_loss[i]=torch.sqrt(stats[0][i])
        # nni.report_intermediate_result({"test/round loss 1": torch.sqrt(stats[0][0]).item(),
        #                                 "test/round loss 2": torch.sqrt(stats[0][1]).item(),
        #                                 "test/round loss 3": torch.sqrt(stats[0][2]).item() ,
        #                                 "test/round loss 4": torch.sqrt(stats[0][3]).item()})
        best_avg_test_loss = (torch.tensor([torch.square(l)*n for l,n in zip(self.client_best_loss,stats[1])])).sum()/(torch.tensor(stats[1]).sum())
        best_avg_test_loss=torch.sqrt(best_avg_test_loss)
        current_avg_test_loss = (torch.tensor([l*n for l,n in zip(stats[0],stats[1])])).sum()/(torch.tensor(stats[1]).sum())
        current_avg_test_loss=torch.sqrt(current_avg_test_loss)
        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("best avg test loss: {:.4f}".format(best_avg_test_loss))
        self.writer.add_scalar("test/best average loss", best_avg_test_loss, round)
        nni.report_intermediate_result({"test/best average loss": best_avg_test_loss.item()})
        print("current avg test loss: {:.4f}".format(current_avg_test_loss))
        self.writer.add_scalar("test/current average loss", current_avg_test_loss, round)
        # nni.report_intermediate_result({"test/average loss": torch.sqrt(stats[0][0]).item()})
        # if self.test_avg_loss>torch.sqrt(total_test_loss).item():
        self.test_avg_loss=best_avg_test_loss.item()
        self.rs_test_rmse.append(self.test_avg_loss)
        print("Test score:" ,stats[2])
        # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test loss: {:.4f}".format(np.std(accs)))
        # print("Std Test score: {:.4f}".format(np.std(aucs)))

    @monitor_gpu_memory
    def test_metrics(self):
        loss_list=[]
        num_samples = []
        score_list = []
        for c in self.clients:
            loss, test_num, score = c.test_metrics()# loss, test_num, score
            loss_list.append(loss)
            num_samples.append(torch.tensor(test_num))
            score_list.append(score)

        return loss_list, num_samples,  score_list

    @monitor_gpu_memory
    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()  # losses, train_num
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses


    @monitor_gpu_memory
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
    @monitor_gpu_memory
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w



    @monitor_gpu_memory
    def RUL_CE(self, n_bins=20, max_rul=125, normalize=True, plot_hist=True):
        rul_ce = []
        all_labels_list = []  # 用于绘图

        for c in self.clients:
            all_labels = []
            for _, labels in c.trainloader:
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                all_labels.append(labels)

            if not all_labels:
                raise ValueError("Dataloader is empty.")

            labels = np.concatenate(all_labels)
            labels = np.clip(labels, 0, max_rul)
            all_labels_list.append(labels)  # 保存用于绘图

            # 分箱
            bin_edges = np.linspace(0, max_rul, n_bins + 1)
            counts, _ = np.histogram(labels, bins=bin_edges)

            total = counts.sum()
            if total == 0:
                entropy_val = 0.0
            else:
                probs = counts / total
                nonzero_probs = probs[probs > 0]
                entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
                if normalize:
                    max_entropy = np.log(n_bins)
                    entropy_val = entropy / max_entropy if max_entropy > 0 else 0.0
                else:
                    entropy_val = entropy
            rul_ce.append(entropy_val)

        print("RUL 分布归一化熵（均匀度）:", rul_ce)

        # ===== 新增：绘制直方图 =====
        if plot_hist and len(self.clients) <= 9:  # 支持最多9个客户端的网格
            n_clients = len(self.clients)
            cols = 2 if n_clients <= 4 else 3
            rows = (n_clients + cols - 1) // cols

            plt.figure(figsize=(5 * cols, 4 * rows))
            bin_edges = np.linspace(0, max_rul, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            for i, labels in enumerate(all_labels_list):
                plt.subplot(rows, cols, i + 1)
                plt.hist(labels, bins=bin_edges, color='skyblue', edgecolor='black', alpha=0.7)
                plt.title(f'Client {i + 1} | Uniformity = {rul_ce[i]:.3f}', fontsize=12)
                plt.xlabel('RUL')
                plt.ylabel('Frequency')
                plt.xlim(0, max_rul)
                plt.grid(axis='y', linestyle='--', alpha=0.6)

            plt.tight_layout()
            plt.show()

        return rul_ce