import os
import torch
import time
import numpy as np
from tqdm import tqdm
from util.logger import logger
from models.DQN import DQN
from models.env import MedicalPlayer

"""
1. 保存模型:使用日志系统,命名系统,tensorboard系统
2. Loss系统进行(根据step)进行优化
"""


class Trainer:
    def __init__(self, viz=True, max_size=3000, max_episodes=100, lr=1e-3, batch_size=16,
                 gamma=0.9, double_dqn=True, train_freq=10, save_freq=50, model_path="../../exp/model/latest.pt",
                 base_path="../data", next_data_freq=20, verbose=False,
                 step_length=1, terminal_step=1e5, padding=10, is_save=False,
                 load_model=False, load_model_path=None, train=True, state_num=2, split_file=None, device=None,
                 gpu=False):
        self.load_model = load_model
        self.is_save = is_save
        self.verbose = verbose
        self.model_path = model_path
        self.viz = viz
        self.save_freq = save_freq
        self.train_freq = train_freq
        self.max_episodes = max_episodes
        self.device = device
        self.env = MedicalPlayer(padding=padding, viz=viz, state_size=(32, 32), step_length=step_length,
                                 base_path=base_path, next_data_freq=next_data_freq, verbose=verbose,
                                 terminal_step=terminal_step, train=train, state_num=state_num, split_file=split_file,
                                 gpu=gpu)
        self.model = DQN(number_actions=self.env.n_actions, max_size=max_size, lr=lr, batch_size=batch_size,
                         gamma=gamma, double_dqn=double_dqn, state_size=(32, 32), state_num=state_num, verbose=verbose)

        if self.load_model and os.path.exists(load_model_path):
            logger.log(f"Loading model from {load_model_path}")
            self.model.eval_net.load_state_dict(torch.load(load_model_path, map_location=device))
            self.model.target_net.load_state_dict(torch.load(load_model_path, map_location=device))
        self.model.eval_net.to(device=device)
        self.model.target_net.to(device=device)

    def train(self):
        acc_step = 0  # 记录智能体决策步骤
        logger.log(f"=== train model with episodes: {self.max_episodes}===")
        for episode in tqdm(range(self.max_episodes)):
            # reset the env,choose_action,step,learn
            state = self.env.reset()
            step_every_episode = 0
            # train:epsilon<1,test:epsilon=1, 训练过程随机探索,epsilon:0-1
            # epsilon = 0 if episode < 50 else min(episode / self.max_episodes, 0.3)
            epsilon = 0
            while True:
                self.env.render()  # 可视化
                # state = state.to(self.device)
                action = self.model.choose_action(state, epsilon)

                next_state, reward, terminal = self.env.step(action)

                if self.verbose:
                    print(f"action={action},reward={reward},terminal={terminal}")

                self.model.store_transition(state, action, reward, next_state)
                if acc_step > 2000 and acc_step % self.train_freq == 0:
                    self.model.learn()
                    self.model.learn()
                state = next_state
                # 中止条件
                if terminal:
                    if self.verbose:
                        logger.log(f"episode={episode},step={step_every_episode}")
                    break
                # 未中止
                acc_step += 1
                step_every_episode += 1
                # if self.viz and episode + 10 > self.max_episodes:
                #     time.sleep(1)
            if self.is_save and episode > 0 and episode % self.save_freq == 0:
                self.save_model(episode)
            # if episode % 10 == 0:
            #     print(reward)
        if self.is_save:
            self.save_model(self.max_episodes)
        if self.viz:
            self.env.mainloop()

    def train_with_two_phase(self):
        acc_step = 0  # 记录智能体决策步骤
        logger.log(f"=== train model with episodes: {self.max_episodes}===")
        for episode in tqdm(range(self.max_episodes)):
            # reset the env,choose_action,step,learn
            state = self.env.reset()
            step_every_episode = 0
            # train:epsilon<1,test:epsilon=1, 训练过程随机探索,epsilon:0-1
            epsilon = 0
            while True:
                self.env.render()  # 可视化
                action = self.model.choose_action(state, epsilon)

                next_state, reward, terminal = self.env.step(action)

                if self.verbose:
                    print(f"action={action},reward={reward},terminal={terminal}")

                self.model.store_transition(state, action, reward, next_state)
                # if acc_step > 1000 and acc_step % self.train_freq == 0:
                #     self.model.learn()
                state = next_state
                # 中止条件
                if terminal:
                    if self.verbose:
                        logger.log(f"episode={episode},step={step_every_episode}")
                    break
                # 未中止
                acc_step += 1
                step_every_episode += 1
                # if self.viz and episode + 10 > self.max_episodes:
                #     time.sleep(1)
            if self.is_save and episode > 0 and episode % self.save_freq == 0:
                self.save_model(episode)
            if episode % 10 == 0:
                print(reward)
        if self.is_save:
            self.save_model(self.max_episodes)
        if self.viz:
            self.env.mainloop()

    def save_model(self, epi):
        path = self.model_path[0:-3] + f"_epi{epi}.pt"
        logger.log(f"save model in {path}")
        torch.save(self.model.eval_net.state_dict(), path)
