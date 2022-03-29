import time
import pickle
import torch
from torch.utils.data import DataLoader
from MyDataset import MyDataset

from utils import calculate_fps
class Detect:
    def __init__(self,img,label):
        self.fps=0.0
        self.error=0
        self.img=img
        self.label=label
        self.dataset = MyDataset(img, label, is_train=False)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.net=torch.load(f'model/99.pt')

    def __call__(self):
        for i,(img,(x_min,x_max,y_min,y_max)) in enumerate(self.dataloader):
            device="cuda:0"
            t1,t2=calculate_fps(img,x_min,x_max,y_min,y_max,device,self.net)
            if t1!=t2:
                # self.fps = (self.fps + (1. / (t2 - t1)))
                self.fps+=1./(t2-t1)
            else:
                self.error+=1
        self.fps/=(len(self.img)-self.error)
        return self.fps,self.error

if __name__ == '__main__':
    test_img = pickle.load(open("../data/train_img.pkl", 'rb'))
    test_label = pickle.load(open("../data/train_label.pkl", 'rb'))
    detect = Detect(test_img, test_label)
    fps,error = detect()
    print(fps)
    print(error)
