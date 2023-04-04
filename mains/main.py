import os
import torch
import time
import datetime
import numpy as np
import pandas as pd
from util.logger import logger  # 导入logger实例
from mains.train.trainer import Trainer
from mains.eval.tester import Tester

"""
训练一个完整模型并保存,测试保存模型效果并保存结果。train->test
version = "V3.4", dataset = "Kits19L", max_episodes = 50, 
"""

# 参数配置
version = "V4"
dataset = "MoNuSeg"  # ["CHAOS-MR(T1)-Liver", "CHAOS-MR(T1)-Spleen", "CHAOS-MR(T1)-LKidney","CHAOS-MR(T1)-RKidney", "CHAOS-CT-Liver", "B5", "B39", "EM", "ssTEM","TNBC","PA", "Kits19L", "Kits19R", "npc1"]
task = "5-shot"  # 固定
max_episodes = 3000
params = {
    # general
    "train": True,  # train or test
    "viz": False,  # if visualize
    "is_save": True,  # if save model after training or testing
    "GPU": False,  # if using GPU
    "verbose": False,  # print more info
    # dataset
    "base_path": fr"D:\datasets\Few-Shot-MedSeg-RL-Based\{dataset}",  # dataset
    "split_file": f'../exp/few-shot/train_ids_{task}-{dataset}.pickle',
    "dataset": dataset,
    "task": task,
    # env
    "step_length": 1,
    "padding": 20,
    "version": version,
    "time": time.strftime("%Y%m%d-%H%M%S", time.localtime()),
    "max_size": 100000,
    # model
    "double_dqn": True,
    "state_num": 4,
    # train
    "max_episodes": max_episodes,
    "terminal_step": 20,
    "batch_size": 32,
    "lr": 0.001,  # 1e-3
    "gamma": 0.9,
    "train_freq": 10,
    "save_freq": 500,
    "next_data_freq": 30,
    "train_result_path": r"../exp/results/train_result.xlsx",
    "load_model": True,
    "model_path": fr"../exp/models/model_{dataset}_{task}_{version}.pt",
    # test
    "type": "random",
    "load_model_path": fr"../exp/models/model_{dataset}_{task}_{version}_epi5000.pt",
    "test_result_path": r"../exp/results/test_result.xlsx",
    # other
    "log_path": r"../exp/logs",
    "comments": "V4: center point"
}
# 使用变量
# 配置代码

if params["GPU"]:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    params["base_path"] = r"/home/sl/data/"

logger.log(f"Loading dataset {dataset} from {params['base_path']}")
logger.log(f"Using params: {params}")

# 逻辑代码
start_time = datetime.datetime.now()
# 建立环境和model
if params["train"]:
    trainer = Trainer(
        viz=params["viz"],
        max_size=params["max_size"],
        max_episodes=params["max_episodes"],
        lr=params["lr"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        double_dqn=params["double_dqn"],
        train_freq=params["train_freq"],
        save_freq=params["save_freq"],
        model_path=params["model_path"],
        base_path=params["base_path"],
        next_data_freq=params["next_data_freq"],
        step_length=params["step_length"],
        terminal_step=params["terminal_step"],
        padding=params["padding"],
        is_save=params["is_save"],
        load_model=params["load_model"],
        train=params["train"],
        split_file=params["split_file"],
        state_num=params["state_num"],
        verbose=params["verbose"],
        load_model_path=params["load_model_path"]
    ).train()
    if params["is_save"]:
        # 结束训练
        end_time = datetime.datetime.now()
        params["train_time(hours)"] = (end_time - start_time).days * 24
        # 保存训练结果
        train_result = pd.read_excel(params["train_result_path"], index_col=0)
        train_result = pd.concat([train_result, pd.DataFrame([params])], ignore_index=True)
        # 记录训练时间
        train_result.to_excel(params["train_result_path"])
else:
    tester = Tester(
        viz=params["viz"],
        base_path=params["base_path"],
        model_path=params["load_model_path"],
        step_length=params["step_length"],
        padding=params["padding"],
        terminal_step=params["terminal_step"],
        train=params["train"],
        split_file=params["split_file"],
        state_num=params["state_num"],
        type=params["type"],
        double_dqn=params["double_dqn"],
        verbose=params["verbose"]
    )
    tester.eval()
    if params["is_save"]:
        # 保存实验指标
        params["mIoU"] = tester.mIoU
        # 保存测试结果
        test_result = pd.read_excel(params["test_result_path"], index_col=0)
        test_result = pd.concat([test_result, pd.DataFrame([params])], ignore_index=True)
        test_result.to_excel(params["test_result_path"])
