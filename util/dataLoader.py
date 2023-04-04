import os
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt

"""
tkinter结合PIL进行图像加载
"""


# 验证信息：图片加载与一致性OK,

def _transform(image, label):
    # 对PIL Image进行变换
    transform = transforms.Compose(
        [
            # 大小
            # transforms.Resize(255), # 缩放
            # transforms.CenterCrop(224), # 裁剪
            # 变换
            # transforms.RandomHorizontalFlip  # 随机旋转
            # 格式
            transforms.ToTensor(),
            # 正则化
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    return transform(image), transform(label)


# 顺序读取文件下的所有文件
def _get_file_names_from_path(file_path):
    assert os.path.exists(file_path), f"{file_path} is not found!"
    file_names = sorted(os.listdir(file_path), key=lambda x: int(x.split(".")[0][12:]))  # listdir读取不是顺序的,设置key:通过文件中数字进行排序
    return [file_path + "/" + file_name for file_name in file_names]


class CellDataset(data.Dataset):
    def __init__(self, image_path, label_path, label_json_path, train=True, split_file=None, crop_size=None):
        """
        image_path, label_path, label_json_path: path of data
        train: train or test
        crop_size: preprocessing
        """
        self.train = train
        self.image_path = image_path
        self.label_path = label_path
        self.label_json_path = label_json_path
        self.crop_size = crop_size
        self.label_json_names = _get_file_names_from_path(label_json_path)
        self.item = 0  # next迭代索引
        self.img_width, self.img_height = 256, 256

        # get image_size
        json_path = self.label_json_names[0]
        label_json = json.load(open(json_path))
        self.img_width, self.img_height = label_json["img_width"], label_json["img_height"]

        # get train or test data with split_file
        train_samples = pickle.load(open(split_file, "rb"))
        if self.train:
            self.label_json_names = np.array(self.label_json_names)[np.array(train_samples)]
        else:
            for idx in train_samples:
                self.label_json_names.pop(idx)

        print(self.label_json_names)

    def __getitem__(self, item):
        label_json_path = self.label_json_names[item]
        name_id = os.path.basename(label_json_path).split(".")[0][12:]
        image_path = self.image_path + fr"/image{name_id}.png"
        label_path = self.label_path + fr"/ground_truth{name_id}.png"

        image = Image.open(image_path)
        if len(image.split()) == 1:
            # 单通道数据->多通道数据
            image = image.convert("RGB")
        label = Image.open(label_path)
        if len(label.split()) == 1:
            label = label.convert("RGB")
        label_json = json.load(open(label_json_path))
        sample = {"Image": image, "Label": label, "Label_json": label_json, "image_path": image_path, "label_path": label_path, "label_json_path": label_json_path}
        return sample

    def __next__(self):
        r = self.__getitem__(item=self.item)
        self.item += 1
        self.item = self.item % len(self)
        return r

    def __len__(self):
        return len(self.label_json_names)


if __name__ == "__main__":
    base_path = r"D:\codes\PaperCodes\FewShotCellSegmentation-master\Datasets\FewShot\Source\B5"
    dataset = CellDataset(image_path=base_path + "/Image", label_path=base_path + "/Groundtruth", label_json_path=base_path + "/Label_Json", train=False,
                          split_file="../../exp/few-shot/train_ids_1-shot-B5.pickle")
    print(dataset[0]["image_path"])
