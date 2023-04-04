import torch
import torch.nn as nn
import numpy as np
from util.logger import logger
from models.expreplay import Memory
from tqdm import tqdm

"""
数据->可视化框架->模型->优化
"""


# 单智能体强化学习
class DQN:
    # The class initialisation function.
    def __init__(self, number_actions=4, lr=1e-3, state_size=(32, 32),
                 max_size=3000, batch_size=16, gamma=0.9, state_num=2,
                 double_dqn=True, verbose=False):
        self.verbose = verbose
        self.double_dqn = double_dqn
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_size = max_size
        self.number_actions = number_actions
        self.lr = lr
        self.state_size = state_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log(f"Using {self.device}")

        self.eval_net, self.target_net = Network2D(number_actions=self.number_actions, state_num=state_num).to(
            self.device), Network2D(number_actions=self.number_actions, state_num=state_num).to(
            self.device)

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        # 记忆体
        self.buffer = Memory(max_size=self.max_size, state_size=self.state_size, state_num=state_num)
        # 使用变量
        self.learn_step_counter = 0
        self.cost = []  # 记录损失值
        self.last_action = 0
        # Freezes target network
        self.target_net.train(False)
        for p in self.target_net.parameters():
            p.requires_grad = False

        # others
        # if self.verbose:
        #     logger.log(self.__dict__)

    def choose_action(self, state, epsilon):
        x = torch.unsqueeze(state, 0)
        if np.random.uniform() <= epsilon:
            action_value = self.eval_net.forward(x)
            # torch.max(action_value,1)提取最大值,[1]->提取值,.data.numpy()->转为numpy,[0]->提取值
            action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
        else:
            action = np.random.randint(0, self.number_actions)
        return action

    def store_batch_transition(self, transition):
        for item in transition:
            self.buffer.append(item[0], item[1], item[2], item[3])

    def store_transition(self, state, action, reward, next_state):
        self.buffer.append(state, action, reward, next_state)
        # # 自学习方式
        # if self.buffer.is_full():
        #     print("learn")
        #     for i in tqdm(range(2000)): self.learn()

    def learn(self):
        if self.learn_step_counter % 100 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_state, sample_action, sample_reward, sample_next_state = self.buffer.sample(self.batch_size)
        # print(sample_state.shape)
        # print(sample_action)
        # print(sample_reward)
        # print(sample_next_state.shape)
        sample_state = sample_state.to(self.device)
        sample_next_state = sample_next_state.to(self.device)
        sample_action = sample_action.to(self.device)
        sample_reward = sample_reward.to(self.device)
        q_eval = self.eval_net(sample_state).gather(1, sample_action)
        q_next_target = self.target_net(sample_next_state).detach()

        if self.double_dqn:
            # double_dqn中q_target计算方法
            q_next_eval = self.eval_net(sample_next_state)
            q_next = q_next_target.gather(1, q_next_eval.max(1)[1].unsqueeze(1))

        else:
            # natural_dqn中q_target计算方法
            q_next = q_next_target.max(1)[0].unsqueeze(1)

        # torch.max->[values=[],indices=[]] max(1)[0]->values=[]
        q_target = sample_reward + self.gamma * q_next.to(self.device)  # Label
        loss = self.loss(q_eval, q_target)
        self.cost.append(loss)
        # 反向传播更新
        self.optimizer.zero_grad()  # 梯度重置
        loss.backward()  # 反向求导
        self.optimizer.step()  # 更新模型参数


class Network2D(nn.Module):
    def __init__(self, number_actions=4, state_num=2):
        super(Network2D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(in_channels=state_num * 3, out_channels=16, kernel_size=(3, 3))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.relu = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))

        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=number_actions)

    def forward(self, x):
        # transforms.ToTensor():numpy的ndarray或PIL.Image读的图片转换成形状为(C,H, W)的Tensor格式，且/255归一化到[0,1.0]之间
        # x = x / 225.0  # [0,1],[batch_size,3,32,32](3@32*32)
        # 1*1卷积进行特征提取
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.relu(self.conv3(x))
        x = self.max_pool(x)  # [batch_size,64,4,4](32@2*2)

        # full connection layers
        x = x.view(x.shape[0], -1)  # 256
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # [batch_size,8]
        return x


if __name__ == "__main__":
    model = DQN()
    r = model.eval_net(torch.rand((32, 3, 32, 32)))
    print(r)
