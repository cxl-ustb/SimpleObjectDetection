from torch.utils.data import Dataset
import cv2
import pickle
import torch
import json
import numpy as np
# 定义数据集的加载方式
class MyDataset(Dataset):
    def __init__(self,dataset,label=None,is_train=False):
        '''
        :param dataset: 数据集的目录
        :param label: 数据集对应的label的目录
        :param is_train: is_train为训练，否则为测试
        '''
        self.dataset=dataset
        self.is_train=is_train
        self.label=label

    def __len__(self):
        '''
        :return: 测试集长度
        '''
        return len(self.dataset)

    def __getitem__(self, idx):
        '''
        :param idx: 取出的图片在DataLoader里面的索引
        :return: 图片信息和目标框拼接后的结果
        '''
        # 取出单张图片
        data=self.dataset[idx]
        img=cv2.imread(data)
        # 将图片转成张量
        new_img=torch.tensor(img).permute(2,0,1)
        json_file = open(self.label[idx])
        # 读取label
        infos = json.load(json_file)
        labels = infos["labels"]
        x_min, x_max, y_min, y_max = labels['agents_labels'][0]["position_in_image"].values()
        x_min, x_max, y_min, y_max = np.float32(x_min), np.float32(x_max), np.float32(y_min),np.float32(y_max)
        return np.float32(new_img),(x_min,x_max,y_min,y_max)


if __name__ == '__main__':
    train_img=pickle.load(open("../data/train_img.pkl",'rb'))
    train_label=pickle.load(open("../data/train_label.pkl",'rb'))
    train_data=MyDataset(train_img,train_label,is_train=True)