import torch
from torch.utils.data import DataLoader
from MyDataset import MyDataset
import pickle
from utils import single_process, calculate_iou,draw_pred_label

device="cuda:0"
class Eval:
    def __init__(self,eval_img,eval_label):
        self.len=len(eval_img)
        # IOU的阈值为0.8
        self.iou_thres=0.8
        # 认为预测的正确图片数
        self.tp=0
        # 认为预测失败的图片数
        self.fp=0
        self.eval_dataset=MyDataset(eval_img,eval_label,is_train=False)
        self.eval_dataloader=DataLoader(self.eval_dataset,batch_size=1,shuffle=False)
        self.net=torch.load(f'model/200.pt')

    def __call__(self):
        for i,(img,(x_min,x_max,y_min,y_max)) in enumerate(self.eval_dataloader):
            example_img,\
            pred_xmin, pred_xmam,\
            pred_ymin, pred_ymax,\
            label_xmin, label_xmax,\
            label_ymin, label_ymax=single_process(img,x_min,x_max,y_min,y_max,device,self.net)
            iou=calculate_iou(pred_xmin,pred_xmam,pred_ymin,pred_ymax,label_xmin,label_xmax,label_ymin,label_ymax)
            #检测预测框和label框iou是否大于阈值
            if iou>=self.iou_thres:
                self.tp+=1
            else:
                # 预测失败就可视化看看区别有多大
                draw_pred_label(example_img, pred_xmin, pred_xmam, pred_ymin, pred_ymax, label_xmin, label_xmax,
                                label_ymin, label_ymax)
                self.fp+=1

        return self.tp/self.len

if __name__ == '__main__':
    test_img = pickle.load(open("../data/train_img.pkl", 'rb'))
    test_label = pickle.load(open("../data/train_label.pkl", 'rb'))
    eval=Eval(test_img,test_label)
    accuracy=eval()
    print(accuracy)