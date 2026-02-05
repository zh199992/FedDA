from distutils.util import strtobool
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
import sys


from system.server.serverCentralized2 import serverCentralized
from system.server.serverFedAvg import serverAvg
from system.server.serverlocal import serverLocal
from system.server.serverGHDR import serverGHDR
from system.server.serverFedDA import serverDA
from system.server.serverFedCADA import serverFedCADA
from system.server.serverFedDA_GHDR import serverDA_GHDR
from system.server.servernewFedCADA import servernewFedCADA
from system.server.serverFedAvgiid import serverAvgiid
from system.server.serverlocaliid import serverLocaliid
from system.server.serverFinetune import serverFinetune
from system.server.serverDANN import serverDANN
from system.server.serverCADA import serverCADA
from models import model
from utils.seed_torch import seed_torch
from utils.gpu_select import get_least_loaded_gpu_id
import threading
import time
import GPUtil

def monitor_gpu_memory(interval=0.5, log_file="gpu_memory.log"):
    with open(log_file, "w") as f:
        f.write("Time(s),GPU_ID,Memory_Used(MB)\n")
    t0 = time.time()
    while True:
        GPUs = GPUtil.getGPUs()
        if GPUs:
            gpu = GPUs[0]  # 假设用 GPU 0
            with open(log_file, "a") as f:
                f.write(f"{time.time() - t0:.2f},{gpu.id},{gpu.memoryUsed}\n")
        time.sleep(interval)


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
            args.model = model.GHDR_FL(data_dim).to(args.device)
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverGHDR(args)  # server初始化用的args包含了新的model
    elif args.algorithm == "local":
        if args.P_FedAvg == True or args.F_FedAvg == True or args.EDI_FedAvg == True:
            sys.exit("不符合local配置")
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        server = serverLocal(args)
    elif args.algorithm == "CADA":
        args.model = model.CADA(lambda_nce=args.lambda_nce).to(args.device)
        if args.mode == "CADA" or args.mode == "wo-InfoNCE" or args.mode == "Source-Only":
            args.aim = args.mode+f"{args.source_id}to{args.target_id}gamma={args.lambda_nce}"
        server = serverCADA(args)
    elif args.algorithm == "DANN":
        if model_str == "cnn1D":
            # args.mode = "baseline"
            # # args.mode = "dann"
            # args.source_id = 1
            # args.target_id = 2

            # if args.EDS:
            #     args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            # else:
            #     args.model = model.GHDR_FL(data_dim).to(args.device) -------------------------------
            args.model = model.conv_DANN(data_dim,args.conv_init,args.gru_init, args.linear_init).to(args.device)
            if args.mode == "dann" or args.mode == "mmd":
                args.aim = args.mode+f"{args.source_id} to {args.target_id} gamma={args.gamma}"
            elif args.mode == "baseline":
                args.aim = args.mode+f"{args.source_id} to {args.target_id}"
            elif args.mode == "mutual":
                args.aim = args.mode+f"{args.source_id} + {args.target_id} gamma={args.gamma}"
            elif args.mode == "centralized":
                args.aim = args.mode+f"target{args.target_id} gamma={args.gamma}"
            elif args.mode == "CADA":
                args.aim = args.mode + f"target{args.target_id} gamma={args.gamma}"
            if args.source_id == args.target_id and args.mode != "centralized":
                raise ValueError("source_id and target_id cannot be the same")
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
    elif args.algorithm == "FedCADA":
        if args.dp != "14-[0,1]":
            raise NotImplementedError
        args.model = model.FedCADA().to(args.device)
        if args.new_da:
            args.server_model = model.Cloud_FedCADA_newda(args.num_clients).to(args.device)
            server = servernewFedCADA(args)
        else:
            args.server_model = model.Cloud_FedCADA(args.num_clients).to(args.device)
            server = serverFedCADA(args)
    elif args.algorithm == "FedDA_GHDR":
        if model_str == "cnn1D":
            if args.EDS:
                args.model = model.GHDR_FL_testeds(data_dim).to(args.device)
            else:
                args.model = model.GHDR_FL(data_dim).to(args.device)
        else:
            raise NotImplementedError
        args.server_model=model.Cloud_GHDR(data_dim,args.window_size, args.num_clients).to(args.device)
        server = serverDA_GHDR(args)
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


    if torch.cuda.is_available():
        # 获取当前默认 GPU 设备（通常是 cuda:0）
        current_device = torch.cuda.current_device()
        print(f"[NNI Trial] Using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
        print(f"[NNI Trial] Total memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB")
    else:
        print("[NNI Trial] Running on CPU")

    # monitor_thread = threading.Thread(target=monitor_gpu_memory, args=(0.1,), daemon=True)
    # monitor_thread.start()

    print(args.model)
    print(args.TIMESTAMP)
    print('modified')
    root_dir = find_project_root('FedDA')
    # directory = '/home/zhouheng/project/FedDA/logs/test1/config/'#比画图多一个/config/
    # directory = 'test2/config/'#比画图多一个/config/
    exp_id = nni.get_experiment_id()
    trial_id = nni.get_trial_id()
    config_path = os.path.join(root_dir, "logs", args.directory, 'config', args.algorithm, args.aim, exp_id,
                               trial_id + '-' + args.TIMESTAMP + '.json')
    graph_path = os.path.join(root_dir, "logs", args.directory, 'graph', args.algorithm, args.aim, exp_id,
                              trial_id + '-' + args.TIMESTAMP)
    index_dir = os.path.join(root_dir, "logs", ".experiment_index")
    os.makedirs(index_dir, exist_ok=True)

    index_file = os.path.join(index_dir, f"{exp_id}.json")

    # 读取已存在的索引文件（如果存在）
    index_data = {}
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            index_data = {}

    # 确保必要的字段存在
    if "experiment_id" not in index_data:
        index_data["experiment_id"] = exp_id

    if "config_paths" not in index_data:
        index_data["config_paths"] = []
    if "graph_paths" not in index_data:
        index_data["graph_paths"] = []

    if "args" not in index_data:
        index_data["args"] = {}

    # 添加新的路径（如果不存在）
    current_config_dir = os.path.join(root_dir, "logs", args.directory, 'config', args.algorithm, args.aim, exp_id)
    current_graph_dir = os.path.join(root_dir, "logs", args.directory, 'graph', args.algorithm, args.aim, exp_id)

    if current_config_dir not in index_data["config_paths"]:
        index_data["config_paths"].append(current_config_dir)
    if current_graph_dir not in index_data["graph_paths"]:
        index_data["graph_paths"].append(current_graph_dir)

    # 更新参数信息
    index_data["args"].update({
        "directory": args.directory,
        "algorithm": args.algorithm,
        "aim": args.aim,
        "TIMESTAMP": args.TIMESTAMP
    })

    # 写入更新后的索引文件
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
#-----------------------------------------------------------------
    print(config_path)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    # yaml_path=os.path.join(directory,  args.algorithm ,args.aim+'.yaml')
    var_args=vars(args)
    var_args = {k: str(v) for k, v in var_args.items()}
    with open(config_path, "w") as file:
        json.dump(var_args, file, indent=4)

    server.train()
    nni.report_final_result(server.test_avg_loss)  # 或者 nni.report_final_result(1 - val_loss)

    # 假设你有验证精度或损失指标
    # 例如 server.best_val_acc 或 server.best_val_loss



if __name__ == '__main__':
    # 在主程序开始前启动监控线程

    # 你的训练代码
    total_start = time.time()
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    # general  添加参数
    parser.add_argument("-git_version", "--git_version", type=str, required=True)
    parser.add_argument("-random_seed", "--random_seed", type=int, default=42)
    parser.add_argument('-d', "--directory", type=str, default="1120")#1021
    parser.add_argument('-aim', "--aim", type=str, default="debug")#训练目的
    parser.add_argument('-data', "--dataset", type=str, default="CMAPSSData")
    parser.add_argument('-m', "--model_name", type=str, default="cnn1D",choices=["cnn1D","lstm","FedRUL"])
    # parser.add_argument('-cloudm', "--model_name", type=str, default="cnn1D",choices=["cnn1D","lstm","FedRUL"])
    parser.add_argument('-dp', "--dp", type=str, default="14-[0,1]",
                        choices=["14-[-1,1]", "18-[0,1]", "14-[0,1]", "18-[-1,1]"], help="dataprocessing")
    parser.add_argument('-train_ratio', "--train_ratio", type=float, default=1.0)
    parser.add_argument('-algo', "--algorithm", type=str, default="CADA",##这个参数不传入server
                        choices=["centralized", "local", "localiid", "FedAvg", "FedPer","FedAvgiid", "GHDR","FedDA","ablation1","ablation2", "finetune", "DANN"])
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
    parser.add_argument('-le2', "--local_epochs2", type=int, default=100, help = "只有GHDR用")
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
    parser.add_argument('-enable_cloud_da', "--enable_cloud_da", type=bool, default=False)
    parser.add_argument('-F_FedAvg', "--F_FedAvg", type=bool, default=False)
    parser.add_argument('-EDI_FedAvg', "--EDI_FedAvg", type=bool,default=False)#不freeze也不fedavg就是个性化
    parser.add_argument('-P_FedAvg', "--P_FedAvg", type=bool,default=False)
    parser.add_argument('-EDI_Freeze', "--EDI_Freeze", type=bool,default=False)
    parser.add_argument('-EDS', "--EDS", type=bool,default=False)#影响模型的forward
    parser.add_argument('-fedeval', "--fedeval", type=bool, default=False)
    parser.add_argument('-DA_loss', type=str, default="adv+mmd",choices=["adv+mmd","adv","mmd","none"])
    parser.add_argument('-lambda_mmd', "--lambda_mmd", type=float, default=0.05)
    parser.add_argument('-gamma', "--gamma", type=float, default=0.05)

    args = parser.parse_args()
#---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    tuned_params = nni.get_next_parameter()
    for key, value in tuned_params.items():
        setattr(args, key, value)
    # ---------------------------------------------------------------------
# ---------------------------------------------------------------------
    if type(args.enable_cloud_da)==str:
        # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.enable_cloud_da = bool(strtobool(args.enable_cloud_da))
    if type(args.new_da) == str:
    # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.new_da = bool(strtobool(args.new_da))
    if type(args.enable_CADA)==str:
        args.enable_CADA = bool(strtobool(args.enable_CADA))
    if type(args.F_FedAvg)==str:
        # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.F_FedAvg = bool(strtobool(args.F_FedAvg))
    if type(args.EDI_FedAvg) == str:
        # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.EDI_FedAvg = bool(strtobool(args.EDI_FedAvg))
    if type(args.P_FedAvg)==str:
        # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.P_FedAvg = bool(strtobool(args.P_FedAvg))
    if type(args.EDI_Freeze)==str:
        # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.EDI_Freeze = bool(strtobool(args.EDI_Freeze))
    if type(args.EDS)==str:
        # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.EDS = bool(strtobool(args.EDS))
    if type(args.fedeval)==str:
        # args.P_FedAvg = bool(strtobool(args.P_FedAvg))
        args.fedeval = bool(strtobool(args.fedeval))

    seed_torch(args.random_seed)
    print("=" * 50) #确认config

    print("Algorithm: {}".format(args.algorithm))
    print("git_version: {}".format(args.git_version))

    print("=" * 50) #确认config

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    args.device = torch.device("cuda")

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

    print("F_FedAvg:{}".format(args.F_FedAvg),type(args.F_FedAvg))
    print("EDI_FedAvg:{}".format(args.EDI_FedAvg))
    print("P_FedAvg:{}".format(args.P_FedAvg),type(args.P_FedAvg))
    print("EDI_Freeze:{}".format(args.EDI_Freeze))
    print("EDS:{}".format(args.EDS))
    print("fedeval:{}".format(args.fedeval),type(args.fedeval))

    print("=" * 50)

    #####输出配置


    run(args)

