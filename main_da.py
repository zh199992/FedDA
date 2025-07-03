import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import adamod
import torch
# import pynvml
import random
import json
import yaml
from datetime import datetime

from system.server.serverCentralized import serverCentralized
from system.server.serverFedAvg import serverAvg
from system.server.serverlocal import serverLocal
from system.server.serverGHDR import serverGHDR
from system.server.serverFedDA import serverDA
from system.server.serverFedAvgiid import serverAvgiid
from system.server.serverlocaliid import serverLocaliid
from models import model
from utils.seed_torch import seed_torch
from utils.gpu_select import get_least_loaded_gpu_id
def run(args):
    model_str = args.model_name
    data_dim = int(args.dp.split('-')[0])

    args.TIMESTAMP=  "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

    # if 'fedeval' in args.aim and args.fedeval:
    #     pass
    # elif 'indeval' in args.aim and not args.fedeval:
    #     pass
    # else:
    #     raise NotImplementedError


    if args.algorithm == "GHDR":
        if model_str == "cnn1D":
            args.model = model.GHDR_FL(data_dim).to(args.device)
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverGHDR(args)  # server初始化用的args包含了新的model
    else:
        raise NotImplementedError

    print(args.model)


    directory = '/home/zhouheng/project/FedDA/logs/da/config/'#比画图多一个/config/
    config_path = os.path.join(directory,  args.algorithm ,args.aim,args.TIMESTAMP+'.json')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    # yaml_path=os.path.join(directory,  args.algorithm ,args.aim+'.yaml')
    var_args=vars(args)
    var_args = {k: str(v) for k, v in var_args.items()}
    with open(config_path, "w") as file:
        json.dump(var_args, file, indent=4)

    server.train()


if __name__ == '__main__':
    seed_torch()

    total_start = time.time()
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # general  添加参数
    parser.add_argument('-aim', "--aim", type=str)#训练目的
    parser.add_argument('-data', "--dataset", type=str, default="CMAPSSData")
    parser.add_argument('-m', "--model_name", type=str, default="cnn1D",choices=["cnn1D","lstm","FedRUL"])
    # parser.add_argument('-cloudm', "--model_name", type=str, default="cnn1D",choices=["cnn1D","lstm","FedRUL"])
    parser.add_argument('-dp', "--dp", type=str, default="18-[0,1]",
                        choices=["14-[-1,1]"  "18-[0,1]"  "14-[0,1]"  "18-[-1,1]"], help="dataprocessing")

    parser.add_argument('-algo', "--algorithm", type=str, default="1",##这个参数不传入server
                        choices=["1"])

    parser.add_argument('-bs_c', "--batch_size_client", type=int, default=256)
    # parser.add_argument('-bs_s', "--batch_size_server", type=int, default=1024)
    # parser.add_argument('-nc', "--num_clients", type=int, default=4,
    #                     help="Total number of clients")
    parser.add_argument('-lr', "--local_learning_rate", type=str, default='0.001,0.001',
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=20)
    parser.add_argument('-le', "--local_epochs", type=str, default='50,5',
                        help="Multiple update steps in one local epoch.")
    # parser.add_argument('-se', "--server_epochs", type=int, default=10)
    # parser.add_argument('-slr', "--server_learning_rate", type=str, default='0.001,0.001')
    # parser.add_argument('-did', "--device_id", type=str, default="0")#?
    # parser.add_argument('-sches', "--server_schedule", type=bool, default=False)
    parser.add_argument('-schec', "--client_schedule", type=bool, default=False)
    # parser.add_argument('-clips', "--server_clip", type=bool, default=False)
    parser.add_argument('-clipc', "--client_clip", type=bool, default=False)
    parser.add_argument('-ws', "--window_size", type=int, default=30)#删了train window还是test window?
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-F_FedAvg', "--F_FedAvg", type=bool, default=False)
    parser.add_argument('-EDI_FedAvg', "--EDI_FedAvg", type=bool,default=False)#不freeze也不fedavg就是个性化
    parser.add_argument('-P_FedAvg', "--P_FedAvg", type=bool,default=False)
    parser.add_argument('-EDI_Freeze', "--EDI_Freeze", type=bool,default=False)
    parser.add_argument('-EDS', "--EDS", type=bool,default=False)#影响模型的forward
    # parser.add_argument('-fedeval', "--fedeval", type=bool, default=False)

    args = parser.parse_args()

    print("=" * 50) #确认config

    print("Algorithm: {}".format(args.algorithm))

    print("=" * 50) #确认config

    gpu_id = get_least_loaded_gpu_id()
    # if gpu_id is not None:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    #     print(f"CUDA_VISIBLE_DEVICES set to {gpu_id}")
    # else:
    #     print("Failed to set CUDA_VISIBLE_DEVICES.")
    args.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')


    print("=" * 50)
    print("Dataset: {}".format(args.dataset))
    print("Algorithm: {}".format(args.algorithm))
    print(f'device{args.device}')
    print(f'dataprocessing{args.dp}')
    print("Backbone: {}".format(args.model_name))
    # print(f'num_clients{args.num_clients}')
    print(f'batch_size_client{args.batch_size_client}')
    # print(f'batch_size_server{args.batch_size_server}')

    print("Local batch size: {}".format(args.batch_size_client))
    # print("server batch size: {}".format(args.batch_size_server))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    # print("Total number of clients: {}".format(args.num_clients))


    print("Global rounds: {}".format(args.global_rounds))

    print("F_FedAvg:{}".format(args.F_FedAvg))
    print("EDI_FedAvg:{}".format(args.EDI_FedAvg))
    print("P_FedAvg:{}".format(args.P_FedAvg))
    print("EDI_Freeze:{}".format(args.EDI_Freeze))
    print("EDS:{}".format(args.EDS))
    # print("fedeval:{}".format(args.fedeval))

    print("=" * 50)

    #####输出配置


    run(args)

