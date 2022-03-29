import torch
from torch import optim
from torch.utils.data import DataLoader
from MyNet import MyNet
from MyDataset import MyDataset
import pickle
import torch.nn as nn

from utils import single_process,draw_pred_label

device="cuda:0"

#虽然类名是Train，但train和test的过程都在这里实现
class Train:
    def __init__(self,train_img,train_label,test_img,test_label,is_train=False):
        '''
        :param train_img: 训练集图片目录
        :param train_label: 训练集label
        :param test_img: 测试集图片目录
        :param test_label: 测试集label目录
        :param is_train:   is_tarin为训练，否则为测试
        '''
        self.train=False
        self.test=False
        if is_train:
            self.train=True
        else:
            self.test=True
        if self.train:
            # 定义训练数据集
            self.train_dataset=MyDataset(train_img, train_label, is_train=True)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=32, shuffle=False)
        else:
            # 定义测试数据集
            self.test_dataset=MyDataset(test_img,test_label,is_train=False)
            self.test_dataloader=DataLoader(self.test_dataset,batch_size=1,shuffle=False)
        # 定义模型
        self.net=MyNet().to(device)
        # 定义优化器
        self.optim=optim.Adam(self.net.parameters())
        # 定义损失函数
        self.position_loss=nn.MSELoss()

    def __call__(self):
        # 训练
        if self.train:
            for epoch in range(1000):
                # 训练过程
                for i,(img,(x_min,x_max,y_min,y_max)) in enumerate(self.train_dataloader):
                    self.net.train()
                    img,(x_min,x_max,y_min,y_max)=img.to(device),(x_min.to(device),x_max.to(device),y_min.to(device),y_max.to(device))
                    pred_position=self.net(img).squeeze(dim=2).squeeze(dim=2)
                    label_position=torch.stack([x_min,x_max,y_min,y_max],dim=0).T
                    loss=self.position_loss(pred_position,label_position)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    print("epoch:{0}".format(epoch))
                    torch.save(self.net,f'model/{epoch}.pt')
        # 测试过程
        if self.test:
            for epoch in range(1):
                for i,(img,(x_min,x_max,y_min,y_max)) in enumerate(self.test_dataloader):
                    # 从训练出来的模型中加载一个
                    self.net=torch.load(f'model/99.pt')
                    # 进行一张图片的检测，返回相应的信息
                    example_img,\
                    pred_xmin, pred_xmam, \
                    pred_ymin, pred_ymax, \
                    label_xmin, label_xmax, \
                    label_ymin, label_ymax = single_process(img, x_min, x_max, y_min, y_max, device, self.net)
                    # 可视化比较label框和预测框
                    draw_pred_label(example_img,pred_xmin,pred_xmam,pred_ymin,pred_ymax,label_xmin,label_xmax,label_ymin,label_ymax)




if __name__ == '__main__':
    # 读取训练/测试集文件名列表
    train_img = pickle.load(open("../data/train_img.pkl", 'rb'))
    train_label = pickle.load(open("../data/train_label.pkl", 'rb'))
    test_img = pickle.load(open("../data/test_img.pkl", 'rb'))
    test_label = pickle.load(open("../data/test_label.pkl", 'rb'))
    # is_train用来决定是进行训练还是测试
    train = Train(train_img, train_label, test_img,test_label,is_train=True)
    # 开始训练或测试
    train()
    pass