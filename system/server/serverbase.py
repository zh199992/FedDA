import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data  #为什么要另外写
import torch.utils.tensorboard as tb
from datetime import datetime

class Server(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.global_model = copy.deepcopy(args.model)###########33
        self.dp=args.dp
        self.batch_size=args.batch_size_server
        self.global_rounds=args.global_rounds
        self.epoches=args.server_epochs
        self.lr=float(args.server_learning_rate.split(',')[0])
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

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        graph_path = '/home/zhouheng/project/FedDA/logs/test1/' + self.algorithm + '/' + TIMESTAMP + self.args.model_name  # localupdateEDI
        self.writer = tb.SummaryWriter(graph_path)  # tensorboard文件存储的文件夹

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

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        # 断言语句 不满足会触发AssertionError
        assert (len(self.clients) > 0)

        active_clients = self.clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []

        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def evaluate(self, round):
        stats = self.test_metrics()
        total_test_loss = (torch.tensor([l*n for l,n in zip(stats[0],stats[1])])).sum()/(torch.tensor(stats[1]).sum())

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test loss: {:.4f}".format(torch.sqrt(total_test_loss)))
        for i in range(len(stats[0])):
            self.writer.add_scalar("test/round loss "+str(i+1), torch.sqrt(stats[0][i]), round)
        self.writer.add_scalar("test/average loss", torch.sqrt(total_test_loss), round)

        print("Test score:" ,stats[2])
        # self.print_(test_acc, train_acc, train_loss)
        # print("Std Test loss: {:.4f}".format(np.std(accs)))
        # print("Std Test score: {:.4f}".format(np.std(aucs)))

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

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w