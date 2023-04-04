"""
1. [threading.Lock()线程锁](https://www.polarxiong.com/archives/python-3-threading-lock.html)
2. [gym](https://www.jianshu.com/p/e136f4fba20c)
"""
import json
import math
import tkinter as tk
from collections import deque

import alphashape
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from shapely.geometry import MultiPolygon, Point
from torchvision import transforms

import util.metric as metric
import util.util as util
from util.dataLoader import CellDataset
from util.logger import logger


# import predict as model


# action V1(current): [shift_x,shift_y],x轴和y轴的偏移量
# action V2: [left,left_top,top,right_top,right,right_bottom,bottom,left_bottom]
# state: 3@512*512,2@320*320,512/360=1.6 1/3区域大小,106.7的范围->
# 16*16和10*10,
# 中止条件: 初始点,单智能体: 一个episode使用一张图片,一张图片terminal后结束，episode设置多余图片数,一张图片就会训练多次
# step:移动具有边界,超出边界移动无效
# reward,terminal方案：当前位置探索下一个点(反复探索)
class MedicalPlayer(tk.Tk):
    def __init__(self, padding=10, viz=False, state_size=(32, 32),
                 step_length=1, base_path="../data", next_data_freq=1,
                 verbose=False, terminal_step=1e5, train=True, state_num=2, type="random",
                 split_file=None, gpu=False):
        """
        viz: visualization,只在一张图片上进行测试
        state_size: 以智能体为中心一定矩形区域作为智能体状态输入到网络中
        """
        super(MedicalPlayer, self).__init__()
        # 参数变量
        self.state_num = state_num
        self.train = train
        self.terminal_step = terminal_step
        self.verbose = verbose
        self.next_data_freq = next_data_freq
        self.viz = viz
        self.padding = padding
        self.action_space = ["left", "up", "right", "down"]
        self.n_actions = len(self.action_space)
        self.state_size = state_size
        self.type = type
        # 使用变量
        self.radius = 2  # 可视化边界点半径
        self.max_step_every_point = 200  # 每一步探索最大500次
        self.step_length = step_length  # 每一次移动像素点步长
        self.reset_seed = 0
        self.res_iou = []
        # 初始逻辑
        # >加载数据
        self.base_path = base_path
        self.dataset = CellDataset(image_path=self.base_path + r"/Image", label_path=self.base_path + r"/Groundtruth",
                                   label_json_path=self.base_path + r"/Label_Json", split_file=split_file,
                                   train=self.train)
        self.current_data = None
        # 一张图片多个对象(环境)
        self.polys = []  # label中边界点
        self.current_poly_index = 0
        self.current_poly = None  # 当前对象,包括box,area,poly
        self.poly = []  # 当前对象中的poly
        self.episode = 0

        # 智能体信息
        # 状态缓存
        self.state_deque = deque(maxlen=state_num)
        # 设置多个智能体
        self.seed_points = []  # 设置多个智能体
        self.seed_index = 0  # 索引
        self.seed_number = 50  # 智能体数可变化训练使用随机选取内部点
        self.skeleton = []  # 所有对象的骨架点
        # 智能体信息
        self.current_pos = (0, 0)  # 当前智能体位置
        self.history_pos = []  # 记录智能体历史位置
        self.curr_his_pos = []  # 当前点历史点
        self.move_step = 0  # 移动步骤
        self.last_pos = (0, 0)  # 上一个位置
        self.result_poly = []
        self.terminal = False
        self.center_point = None  # 中心点
        self.img_width, self.img_height = self.dataset.img_width, self.dataset.img_height
        open_file = fr"D:\datasets\Cell\MoNuSeg\val\points\{self.type}.json"
        if gpu:
            open_file = f"/home/sunl/data/MoNuSeg/val/points/{self.type}.json"
        with open(open_file) as f:
            data = json.load(f)
            self.images = data["Images"]
        self.cur_images_index = 0
        if self.viz:
            self._build_screen()
        else:
            self.destroy()

        if self.verbose:
            logger.log("env param: " + str(self.__dict__))

    # for train
    def reset(self):
        # 加载环境 -> 随机生成多个初始交互点 -> 返回当前状态
        # >加载下一张图片,self.next_data_freq * len(self.polys): 每张图片有多个poly, 完成一个图片的所有poly后加载下一张图片
        if self.episode == 0 or self.episode % len(self.polys) == 0:
            self.current_data = next(self.dataset)
            self.current_poly_index = 0
            self.polys = self.current_data["Label_json"]["polys"]
        else:
            self.current_poly_index += 1
            self.current_poly_index = self.current_poly_index % len(self.polys)

        # 初始化一些使用变量
        self.current_poly = self.polys[self.current_poly_index]
        # 轮廓外接box,box中心点作为初始点
        box = self.current_poly["box"]  # [[],[]],左上和右下坐标
        self.poly = self.current_poly["poly"]
        self.center_point = util.get_center_point(np.array(self.poly))
        # self._draw_position(self.center_point, fill='blue')
        # 生成种子点: 内部种子点
        self.seed_index = 0
        self.seed_points.clear()
        for i in range(self.seed_number):
            seed = (np.random.randint(box[0][0], box[1][0]), np.random.randint(box[0][1], box[1][1]))
            while self._get_polygon_dist_from_point(seed) < 0:
                seed = (np.random.randint(box[0][0], box[1][0]), np.random.randint(box[0][1], box[1][1]))
            self.seed_points.append(seed)

        self.seed_number = len(self.seed_points)
        self.current_pos = self.seed_points[self.seed_index]
        self.terminal = False
        self.last_pos = self.current_pos
        self.history_pos.clear()
        self.history_pos.append(list(self.current_pos))
        self.curr_his_pos.clear()
        self.curr_his_pos.append(list(self.current_pos))
        self.move_step = 0
        self.img_width, self.img_height = self.current_data["Label_json"]["img_width"], self.current_data["Label_json"][
            "img_height"]

        # >返回初始状态
        self.state_deque.clear()
        state = self._get_state_from_position(self.current_pos)
        self.episode += 1

        if self.viz:
            # 初始化智能体位置并显示
            self._reset_screen()
            self._draw_position(self.current_pos)

        return state

    def step(self, action):
        reward = self._update_current_pos(action)  # 更新步骤, 并切状态, 并判断中止条件terminal
        next_state = self._get_state_from_position(self.current_pos)
        # terminal = self._get_terminal_with_move_step()
        return next_state, reward, self.terminal

    def _update_current_pos(self, action):
        # history_pos更新,current_pos更新
        next_pos = self._get_next_pos_from_action(action)
        dist = self._get_polygon_dist_from_point(next_pos)
        # reward
        reward = self._calc_distance(self.center_point, self.current_pos) - self._calc_distance(self.center_point,
                                                                                                next_pos)
        # reward = dist

        if next_pos not in self.curr_his_pos:
            self.curr_his_pos.append(list(next_pos))
        if self.move_step + 1 >= self.terminal_step or dist < 0:
            if self.seed_index >= self.seed_number:
                self.terminal = True
                self.current_pos = next_pos  # 用于对齐可视化逻辑
            else:
                self.current_pos = self.seed_points[self.seed_index]
                self.move_step = 0
                self.seed_index += 1
                self.state_deque.clear()
                self.curr_his_pos.clear()
                self.curr_his_pos.append(list(self.current_pos))

        else:
            self.last_pos = self.current_pos
            self.current_pos = next_pos
            self.move_step += 1

        if self.viz:
            self._draw_position(self.current_pos)
            self._draw_position(self.center_point, fill='blue')
        return round(reward, 2)

    # for eval
    def reset_eval(self, img_path, point):
        # 加载环境 -> 模拟用户提供交互点 -> 返回当前状态
        self.current_data = {
            "Image": Image.open(img_path)
        }
        self.current_pos = point
        self.terminal = False
        self.last_pos = self.current_pos
        self.history_pos.clear()
        self.history_pos.append(list(self.current_pos))
        self.curr_his_pos.clear()
        self.curr_his_pos.append(list(self.current_pos))
        self.move_step = 0
        # >返回初始状态
        self.state_deque.clear()
        state = self._get_state_from_position(self.current_pos)
        self.episode += 1

        if self.viz:
            # 初始化智能体位置并显示
            self.canvas.delete(tk.ALL)
            img_pil = Image.open(img_path)
            self.photo_image = ImageTk.PhotoImage(img_pil)
            self.img_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
            self._draw_position(self.current_pos)
            self.canvas.pack()

        return state

    def step_eval(self, action):
        # self.step_length = np.random.randint(1, 5)
        self._update_current_pos_for_eval(action)
        next_state = self._get_state_from_position(self.current_pos)
        # print(f"step_eval,action={action},pos={self.current_pos}")
        return next_state, self.terminal

    def _update_current_pos_for_eval(self, action):
        # history_pos更新,current_pos更新
        next_pos = self._get_next_pos_from_action(action)
        if list(next_pos) not in self.history_pos:
            # print(f"next {next_pos}")
            self.history_pos.append(list(next_pos))  # 放在self.current_pos更新后,不记录初始位置
        # or self.last_pos == next_pos 震荡判断: 只有不使用历史状态才有这样的震荡状态
        # if self.move_step >= self.terminal_step or next_pos == self.last_pos:
        if self.move_step >= self.terminal_step:
            self.terminal = True
        else:
            self.last_pos = self.current_pos
            self.current_pos = next_pos
            self.move_step += 1

        # 可视化
        if self.viz:
            self._draw_position(self.current_pos)

    def get_iou(self, poly):
        # alphashape: 设置alpha运行速度快,不设置alpha自行搜索最佳方案,速度慢,alpha_shape:Polygon对象
        try:
            alpha_shape = alphashape.alphashape(poly, 0.08)
            if isinstance(alpha_shape, MultiPolygon):
                alpha_shape = alphashape.alphashape(poly, 0.)
        except:
            return 0
        if isinstance(alpha_shape, Point):
            return 0
        polygon = list(alpha_shape.exterior.coords)

        # 测试提取边界结果
        # fig, ax = plt.subplots()
        # ax.scatter(*zip(*self.result_poly))
        # ax.add_patch(PolygonPatch(alpha_shape, alpha=0.9))
        # ax.invert_yaxis()
        # plt.show()
        if self.viz:
            self.canvas.create_polygon(polygon, outline="red", width=self.radius, fill="")
            self.canvas.pack()
        return util.calc_iou_with_polygon(np.array(polygon), np.array(self.poly))

    def get_polygon_from_gather(self, points):
        try:
            alpha_shape = alphashape.alphashape(points, 0.08)
            if isinstance(alpha_shape, MultiPolygon):
                alpha_shape = alphashape.alphashape(points, 0.)
        except:
            return []
        if isinstance(alpha_shape, Point):
            return []
        polygon = list(alpha_shape.exterior.coords)
        return polygon

    def get_iou_from_polygon(self):
        self.result_poly = self.history_pos
        return self.get_iou(self.result_poly)

    # for all
    def is_action_available(self, action):
        # 保证action符合实际情况:不超出搜索边界
        box = self.current_poly["box"]
        left, top = max(box[0][0] - self.padding, 0), max(box[0][1] - self.padding, 0)
        right, bottom = min(box[1][0] + self.padding, self.img_width - 1), min(box[1][1] + self.padding,
                                                                               self.img_height - 1)
        x, y = self._get_next_pos_from_action(action)
        if left <= x <= right and top <= y <= bottom:
            return True
        else:
            return False

    def post_process_with_grab_cut(self):
        # grabcut： self.history_pos点集合中提取出 bbox 和 先验 -> GrabCut -> 结果
        contour = np.array(self.history_pos)
        lef_top = contour.min(axis=0)
        right_bottom = contour.max(axis=0)
        box = util.add_box_padding((self.img_height, self.img_width), [lef_top, right_bottom], self.padding)  # √
        # GrabCut
        mask = np.zeros((self.img_height, self.img_width), np.uint8)
        poly = np.array(self.get_polygon_from_gather(self.history_pos), np.int)
        # util.show_cv_img(mask)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (box[0][0], box[0][1], box[1][0] - box[0][0], box[1][1] - box[0][1])  # (x,y,w,h)
        img = util.pil_to_cv2(self.current_data["Image"])

        # rect
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=1, mode=cv2.GC_INIT_WITH_RECT)
        # 调用GC_INIT_WITH_MASK前需要先调用GC_INIT_WITH_RECT
        # for i in range(box[0][0], box[1][0]):
        #     for j in range(box[0][1], box[1][1]):
        #         if self._get_polygon_dist((i, j), poly) >= 0:
        #             mask[j][i] = 1
        res_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        res_mask[res_mask == 1] = 255
        print("rect=",
              metric.mask_iou(metric.get_mask(res_mask), metric.get_mask(util.pil_to_cv2(self.current_data["Label"]))))
        # util.show_cv_img(res_mask)

        # mask
        for point in self.history_pos:
            mask[point[1]][point[0]] = 1
        # util.show_cv_img(mask)
        mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
        # except:
        #     print(f"can't rect seg")
        res_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        res_mask[res_mask == 1] = 255
        util.show_cv_img(res_mask)

        # result
        iou = metric.mask_iou(metric.get_mask(res_mask), metric.get_mask(util.pil_to_cv2(self.current_data["Label"])))
        self.res_iou.append(iou)
        print("mask = ", iou)
        print("mean post res = ", round(float(np.mean(self.res_iou)), 4))
        contours = util.get_contour_with_mask(res_mask)
        post_mask = np.zeros(res_mask.shape, dtype=np.uint8)
        cv2.fillPoly(post_mask, contours, (255, 255, 255))
        print(metric.mask_iou(metric.get_mask(post_mask), metric.get_mask(util.pil_to_cv2(self.current_data["Label"]))))

        if self.viz:
            self.canvas.create_rectangle(box[0][0], box[0][1], box[1][0], box[1][1])

    def post_process_with_cnn(self, method="MIDeepseg"):
        # if method == "MIDeepseg":
            # self.seed_points 和 self.history_pos 分别输入可以得到对比结果
            # contour = model.predict_with_mideepseg(self.current_data["image_path"], self.seed_points.copy())
            # contour = contour.squeeze()
        if method == "DEXTR":
            contour = []
            print()

        if self.viz:
            self.canvas.create_polygon(contour.tolist(), outline="red", width=self.radius, fill="")
            self.canvas.pack()

        # 计算结果
        self.res_iou.append(util.calc_iou_with_polygon(np.array(contour), np.array(self.poly)))
        return contour

    def _get_terminal_with_move_step(self):
        if self.move_step >= self.terminal_step and self.seed_index + 1 >= self.seed_number:
            # print(f"terminal,move_step={self.move_step},seed_index={self.seed_index}")
            return True
        else:
            return False

    def _is_position_available(self, pos):
        if 0 <= pos[0] < self.img_width and 0 <= pos[1] < self.img_height:
            return True
        else:
            return False

    # for util
    def _get_reward(self, action):
        next_pos = self._get_next_pos_from_action(action)
        # 驱动智能体在内部运动 : 1) 越接近边界收益越大, 2) 超出边界直接terminal 3) 走过的路径 0.5 * reward
        dist = self._get_polygon_dist_from_point(next_pos)
        if dist > 0:
            # if next_pos == self.last_pos:
            #     reward = 0
            # else:
            reward = 1 / dist
        else:
            reward = dist
        return round(reward, 2)

    def _get_polygon_dist_from_point(self, point):
        """计算点到多边形轮廓距离"""
        contour = np.array(self.poly)
        return self._get_polygon_dist(point, contour)

    def _get_polygon_dist(self, point, contour):
        dist = cv2.pointPolygonTest(contour, point, measureDist=True)
        return round(dist, 2)

    def _build_screen(self):
        # tk配置信息
        self.title("seg")
        self.geometry("+700+300")
        # 创建画布
        self.canvas = tk.Canvas(self, bg="white", height=self.img_height, width=self.img_width)
        # img_pil = self.current_data["Image"]
        # self.photo_image = ImageTk.PhotoImage(img_pil)
        # self.img_canvas = self.canvas.create_image(250, 0, anchor="n", Image=self.photo_image)
        # self.canvas.pack()

    def _reset_screen(self):
        self.canvas.delete(tk.ALL)
        img_pil = self.current_data["Image"]
        self.photo_image = ImageTk.PhotoImage(img_pil)
        self.img_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.create_polygon(self.poly, outline="green", width=self.radius, fill="")
        box = self.current_poly["box"]
        self.canvas.create_rectangle(box[0][0], box[0][1], box[1][0], box[1][1])
        self.canvas.pack()

    def _get_state_from_position(self, position):
        img_pil = self.current_data["Image"]
        img_crop = img_pil.crop((position[0] - self.state_size[0] / 2, position[1] - self.state_size[1] / 2,
                                 position[0] + self.state_size[0] / 2, position[1] + self.state_size[1] / 2))
        # test img_crop
        # plt.figure()
        # plt.imshow(img_crop)
        # plt.show()
        # get_recent_state
        state = transforms.ToTensor()(img_crop)  # 3*32*32
        # 填充满(初始状态时没有历史状态)-> 与历史状态连接
        while len(self.state_deque) < self.state_deque.maxlen:
            self.state_deque.append(state)
        self.state_deque.append(state)
        recent_state = torch.cat(tuple([s for s in self.state_deque]), 0)
        return recent_state

    def _get_next_pos_from_action(self, action):
        # ["left", "up", "right", "down"]
        x, y = self.current_pos[0], self.current_pos[1]
        if action == 0:
            x -= self.step_length
        if action == 1:
            y -= self.step_length
        if action == 2:
            x += self.step_length
        if action == 3:
            y += self.step_length
        return x, y

    def _get_nexts_using_center_point(self, center_point):
        # ["left", "up", "right", "down"]
        res = []
        step = self.step_length
        res.append(center_point)
        res.append((center_point[0] - step, center_point[1]))
        res.append((center_point[0] + step, center_point[1]))
        res.append((center_point[0], center_point[1] - step))
        res.append((center_point[0], center_point[1] + step))
        res.append((center_point[0] - step, center_point[1] - step))
        res.append((center_point[0] - step, center_point[1] + step))
        res.append((center_point[0] + step, center_point[1] - step))
        res.append((center_point[0] + step, center_point[1] + step))
        return res

    def _get_points_using_extremes_points(self, dist=10):
        # 极值点 => 首尾10像素范围生成=> 满足在内
        extreme_points = util.get_extreme_points(self.poly)
        res = []
        for point in extreme_points:
            w = 20  # 圆平均分为20份
            m = (2 * math.pi) / w  # 一个圆分成10份，每一份弧度为 m
            for i in range(0, w + 1):
                x = point[0] + dist * math.sin(m * i)
                y = point[1] + dist * math.cos(m * i)
                if self._get_polygon_dist_from_point((x, y)) > 0:
                    res.append((int(x), int(y)))
                    break
        return res

    def render(self):
        if self.viz:
            self.update()

    @staticmethod
    def _calc_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def get_distance(self, point1, point2):
        return self._calc_distance(point1, point2)

    def _draw_position(self, position, fill="red"):
        self.canvas.create_oval(position[0] - self.radius, position[1] - self.radius, position[0] + self.radius,
                                position[1] + self.radius, fill=fill)  # x0,y0,x1,y2:外接矩形的左上和右下坐标
        self.canvas.pack()
