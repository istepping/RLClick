import json
import os
import torch
import numpy as np
import time
from util.logger import logger
from models.DQN import DQN
from models.env import MedicalPlayer
from tqdm import tqdm

"""
做数据调试以及验证
"""


class Tester:
    def __init__(self, viz=True, base_path="../data",
                 max_size=3000, lr=0.001, batch_size=16,
                 gamma=0.9, double_dqn=True, model_path="../exp/model/latest.pt",
                 step_length=1, verbose=False, padding=10, train=False, type="random",
                 split_file=None, state_num=2, terminal_step=5):
        self.verbose = verbose
        self.viz = viz
        self.type = type
        self.double_dqn = double_dqn
        self.env = MedicalPlayer(padding=padding, viz=viz, state_size=(32, 32),
                                 step_length=step_length, base_path=base_path, next_data_freq=1,
                                 verbose=verbose, train=train, split_file=split_file, state_num=state_num,
                                 terminal_step=terminal_step, type=type)
        self.model = DQN(number_actions=self.env.n_actions, max_size=max_size, lr=lr, batch_size=batch_size,
                         gamma=gamma, double_dqn=double_dqn, state_size=(32, 32), state_num=state_num, verbose=verbose)
        # return param: -1 has not been calculated
        self.iou_list = []
        self.mIoU = -1
        logger.log(f"Loading model from {model_path}")
        self.model.eval_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def eval(self):
        with open(fr"D:\datasets\Cell\MoNuSeg\val\points\center.json") as f:
            center_data = json.load(f)
            center_images = center_data["Images"]

        with open(fr"D:\datasets\Cell\MoNuSeg\val\points\{self.type}.json") as f:
            data = json.load(f)
            images = data["Images"]
        n_data = len(images)
        logger.log(f"test image num: {n_data}")
        img_info = dict()
        if self.double_dqn:
            img_info["Type"] = self.type + "_DDQN"
        else:
            img_info["Type"] = self.type + "_DQN"
        imgs = []
        all_time = []
        for i in tqdm(range(n_data)):
            image = dict()
            image["img_path"] = images[i]["img_path"]
            image["mask_path"] = images[i]["mask_path"]
            n_item = len(images[i]["points"])
            points = []
            center_points = center_images[i]["points"]
            for j in range(n_item):
                time1 = time.time()
                state = self.env.reset_eval(images[i]["img_path"], images[i]["points"][j])
                while True:
                    self.env.render()
                    action = self.model.choose_action(state, epsilon=1)
                    next_state, terminal = self.env.step_eval(action)
                    state = next_state
                    if terminal:
                        break
                # 后处理: 记录位置, 保存结果
                time2 = time.time()
                all_time.append(time2 - time1)
                points.append(self.env.current_pos)
                self.iou_list.append(self.env.get_distance(self.env.current_pos, center_points[j]))
                self.env.render()
                if self.viz:
                    time.sleep(2)
            image["points"] = points
            imgs.append(image)

        img_info["Images"] = imgs
        base_path = r"D:\datasets\Cell\MoNuSeg\val"
        with open(base_path + fr"\points\{img_info['Type']}.json", "w") as f:
            json.dump(img_info, f, indent=3)
        print("save points in " + fr"\points\{img_info['Type']}.json")
        # 测试环境保存预测结果poly-后续计算评分
        self.mIoU = round(float(np.mean(self.iou_list)), 4)
        logger.log("mean distance=" + str(self.mIoU))
        logger.log(f"mean time={np.mean(np.array(all_time))}")
        # 可视化主循环
        if self.viz:
            self.env.mainloop()


"""
1. 产生交互点, 保存结果。
2. 重新训练模型。
"""
