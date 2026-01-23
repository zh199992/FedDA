import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from utils.data_utils import str_to_bool
from utils.root import find_project_root
import nni


from system.server.serverCentralized2 import serverCentralized
from system.server.serverFedAvg import serverAvg
from system.server.serverlocal import serverLocal
from system.server.serverGHDR import serverGHDR
from system.server.serverFedDA import serverDA
from system.server.serverFedAvgiid import serverAvgiid
from system.server.serverlocaliid import serverLocaliid
from system.server.serverFinetune import serverFinetune
from system.server.serverDANN import serverDANN
from models import model
from utils.seed_torch import seed_torch
from utils.gpu_select import get_least_loaded_gpu_id
def run(args):
    model_str = args.model_name
    data_dim = int(args.dp.split('-')[0])

    args.TIMESTAMP=  "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

    if 'fedeval' in args.aim and args.fedeval:
        pass
    elif 'indeval' in args.aim and not args.fedeval:
        pass
    elif not 'indeval' in args.aim and not 'fedeval' in args.aim:
        pass
    else:
        raise NotImplementedError


    if args.algorithm == "GHDR":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverGHDR(args)  # server初始化用的args包含了新的model
    elif args.algorithm == "all":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverGHDR(args)

    elif args.algorithm == "local":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverLocal(args)
    elif args.algorithm == "DANN":
        if model_str == "cnn1D":
            # args.mode = "baseline"
            # # args.mode = "dann"
            # args.source_id = 1
            # args.target_id = 2
            if args.source_id == args.target_id:
                raise ValueError("source_id and target_id cannot be the same")
            # if args.EDS:
            #     args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            # else:
            #     args.model = model.GHDR_FL(data_dim).to(args.device) -------------------------------
            args.model = model.conv_DANN(data_dim,args.conv_init,args.gru_init, args.linear_init).to(args.device)
            if args.mode == "dann":
                args.aim = args.mode+f"{args.source_id} to {args.target_id} gamma={args.gamma}"
            elif args.mode == "baseline":
                args.aim = args.mode+f"{args.source_id} to {args.target_id}"
        else:
            raise NotImplementedError
        server = serverDANN(args)
    elif args.algorithm == "finetune":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverFinetune(args)
    elif args.algorithm == "localiid":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverLocaliid(args)
    elif args.algorithm == "FedAvg":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        # elif model_str == "cnn2D":
        else:
            raise NotImplementedError
        server = serverAvg(args)
    elif args.algorithm == "FedDA":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim,args.conv_init,args.gru_init, args.linear_init).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim,args.conv_init,args.gru_init, args.linear_init).to(args.device)
                # args.model = model.GHDR_FL_new(data_dim,args.conv_init,args.gru_init, args.linear_init).to(args.device)
        else:
            raise NotImplementedError
        args.server_model=model.Cloud_GHDR(data_dim,args.window_size, args.num_clients,args.conv_init, args.linear_init).to(args.device)
        # args.server_model=model.Cloud_GHDR_new(data_dim,args.window_size, args.num_clients,args.conv_init, args.linear_init).to(args.device)
        server = serverDA(args)
    elif args.algorithm == "FedAvgiid":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverAvgiid(args)
    elif args.algorithm == "centralized":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverCentralized(args)
    else:
        raise NotImplementedError

    print(args.model)
    print(args.TIMESTAMP)
    print('modified')
    root_dir = find_project_root('FedDA')
    # directory = '/home/zhouheng/project/FedDA/logs/test1/config/'#比画图多一个/config/
    # directory = 'test2/config/'#比画图多一个/config/
    # exp_id = nni.get_experiment_id()
    # trial_id = nni.get_trial_id()
    config_path = os.path.join(root_dir, "logs", args.directory, 'config', args.algorithm ,args.aim, args.TIMESTAMP+'.json')
    graph_path = os.path.join(root_dir, "logs", args.directory, 'graph', args.algorithm, args.aim, args.TIMESTAMP)
    index_dir = os.path.join(root_dir, "logs", ".experiment_index")
    os.makedirs(index_dir, exist_ok=True)

    # 写入索引：用 experiment_id 作为文件名，内容包含 graph 和 config 路径
#-----------------------------------------------------------------
    print(config_path)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    # yaml_path=os.path.join(directory,  args.algorithm ,args.aim+'.yaml')
    var_args=vars(args)
    var_args = {k: str(v) for k, v in var_args.items()}
    with open(config_path, "w") as file:
        json.dump(var_args, file, indent=4)

    server.train()
    # 假设你有验证精度或损失指标
    # 例如 server.best_val_acc 或 server.best_val_loss



if __name__ == '__main__':

    total_start = time.time()
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # general  添加参数
    parser.add_argument("-random_seed", "--random_seed", type=int, default=42)
    parser.add_argument('-d', "--directory", type=str, default="1021")
    parser.add_argument('-aim', "--aim", type=str, default="debug")#训练目的
    parser.add_argument('-data', "--dataset", type=str, default="CMAPSSData")
    parser.add_argument('-m', "--model_name", type=str, default="cnn1D",choices=["cnn1D","lstm","FedRUL"])
    # parser.add_argument('-cloudm', "--model_name", type=str, default="cnn1D",choices=["cnn1D","lstm","FedRUL"])
    parser.add_argument('-dp', "--dp", type=str, default="18-[0,1]",
                        choices=["14-[-1,1]", "18-[0,1]", "14-[0,1]", "18-[-1,1]"], help="dataprocessing")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg",##这个参数不传入server
                        choices=["centralized", "local", "localiid", "FedAvg", "FedAvgiid", "GHDR","FedDA","ablation1","ablation2", "finetune", "DANN", "all"])
    parser.add_argument('-o_c', "--optimizer_client", type=str, default="adam", choices=["adam", "adamod", "sgd"])
    parser.add_argument('-o_s', "--optimizer_server", type=str, default="adam", choices=["adam", "adamod", "sgd"])
    parser.add_argument('-bs_c', "--batch_size_client", type=int, default=1024)
    parser.add_argument('-bs_s', "--batch_size_server", type=int, default=1024)
    parser.add_argument('-nc', "--num_clients", type=int, default=4,
                        help="Total number of clients")
    parser.add_argument('-soft_update', "--soft_update", type=bool, default=False)#我想到T/F可以用0/1来表示
    parser.add_argument('-miu_su', "--miu_su", type=float, default=0.05)
    parser.add_argument('-gr_i', "--global_rounds_init", type=int, default=0)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)

    parser.add_argument('-early_stop', "--early_stop", type=bool, default=True)
    parser.add_argument('-pretrain_early_stop', "--pretrain_early_stop", type=bool, default=True)
    # parser.add_argument('-le', "--local_epochs", type=str, default='50,5',
    #                     help="Multiple update steps in one local epoch.")
    parser.add_argument('-le', "--local_epochs", type=int, default=100,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-se', "--server_epochs", type=int, default=10)
    # parser.add_argument('-clr', "--local_learning_rate", type=str, default='0.001,0.001')
    # parser.add_argument('-slr', "--server_learning_rate", type=str, default='0.001,0.001')
    parser.add_argument('-clr', "--local_learning_rate", type=float, default=0.001)
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.001)
    # parser.add_argument('-did', "--device_id", type=str, default="0")#?
    parser.add_argument('-sches', "--server_schedule", type=bool, default=False)
    parser.add_argument('-schec', "--client_schedule", type=bool, default=False)
    parser.add_argument('-clips', "--server_clip", type=bool, default=False)
    parser.add_argument('-clipc', "--client_clip", type=bool, default=False)
    parser.add_argument('-ws', "--window_size", type=int, default=30)#删了train window还是test window?
    parser.add_argument('-lrd_c', "--client_lr_decay", type=bool, default=False)
    parser.add_argument('-lrd_s', "--server_lr_decay", type=bool, default=False)

    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-conv_init', "--conv_init", type=str, default="kaiming_uniform",
                        choices=["kaiming_uniform","kaiming_normal","xavier_uniform","xavier_normal","normal","uniform"])
    parser.add_argument('-gru_init', "--gru_init", type=str, default="kaiming_uniform",
                        choices=["kaiming_uniform","kaiming_normal","xavier_uniform","xavier_normal","normal","uniform"])
    parser.add_argument('-linear_init', "--linear_init", type=str, default="kaiming_uniform",
                        choices=["kaiming_uniform","kaiming_normal","xavier_uniform","xavier_normal","normal","uniform"])
    parser.add_argument('-F_FedAvg', "--F_FedAvg", type=bool, default=False)
    parser.add_argument('-EDI_FedAvg', "--EDI_FedAvg", type=bool,default=False)#不freeze也不fedavg就是个性化
    parser.add_argument('-P_FedAvg', "--P_FedAvg", type=bool,default=False)
    parser.add_argument('-EDI_Freeze', "--EDI_Freeze", type=bool,default=False)
    parser.add_argument('-EDS', "--EDS", type=bool,default=False)#影响模型的forward
    parser.add_argument('-fedeval', "--fedeval", type=bool, default=False)
    parser.add_argument('-DA_loss', type=str, default="adv+mmd",choices=["adv+mmd","adv","mmd","none"])
    parser.add_argument('-lambda_mmd', "--lambda_mmd", type=float, default=0.05)
    parser.add_argument('-gamma', "--gamma", type=float, default=0.05)


#---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    parser.add_argument('-source_id', "--source_id", type=int, default=1)
    parser.add_argument('-target_id', "--target_id", type=int, default=2)
    parser.add_argument('--mode', "--mode", type=str, default='dann')
    # ---------------------------------------------------------------------
# ---------------------------------------------------------------------
    args = parser.parse_args()
    seed_torch(args.random_seed)
    print("=" * 50) #确认config

    print("Algorithm: {}".format(args.algorithm))

    print("=" * 50) #确认config

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    args.device = torch.device("cuda:1")

    print("=" * 50)
    print("random_seed: {}".format(args.random_seed))
    print("Dataset: {}".format(args.dataset))
    print("Algorithm: {}".format(args.algorithm))
    print(f'device{args.device}')
    print(f'dataprocessing{args.dp}')
    print("Backbone: {}".format(args.model_name))
    print(f'num_clients{args.num_clients}')
    print("Local batch size: {}".format(args.batch_size_client))
    print("server batch size: {}".format(args.batch_size_server))
    print("Local epochs: {}".format(args.local_epochs))
    print("Server epochs: {}".format(args.server_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Server learing rate: {}".format(args.server_learning_rate))
    print("client learing rate decay: {}".format(args.client_lr_decay))
    print("Server learing rate decay: {}".format(args.server_lr_decay))
    if args.client_lr_decay or args.server_lr_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))


    print("Global rounds: {}".format(args.global_rounds))

    print("F_FedAvg:{}".format(args.F_FedAvg))
    print("EDI_FedAvg:{}".format(args.EDI_FedAvg))
    print("P_FedAvg:{}".format(args.P_FedAvg))
    print("EDI_Freeze:{}".format(args.EDI_Freeze))
    print("EDS:{}".format(args.EDS))
    print("fedeval:{}".format(args.fedeval))

    print("=" * 50)

    #####输出配置


    run(args)

