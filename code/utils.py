import torch
import numpy as np
import cv2
import time
# 计算1s的时间能够推理多少张图片
def calculate_fps(img,x_min,x_max,y_min,y_max,device,net):
    # 推理开始时间
    t1=time.time()
    img, (x_min, x_max, y_min, y_max) = img.to(device), (x_min.to(device), x_max.to(device), y_min.to(device), y_max.to(device))
    pred_position = net(img).squeeze(dim=2).squeeze(dim=2)
    # 推理结束时间
    t2=time.time()
    # 返回推理开始，结束时间
    return t1,t2

# 抽取图片的label信息，使用模型对图片进行检测
def single_process(img,x_min,x_max,y_min,y_max,device,net):
    '''
    :param img: 单张图片的数据
    :param x_min: label x_min
    :param x_max: label x_max
    :param y_min: label y_min
    :param y_max: label y_max
    :param device:
    :param net: 检测网络
    :return:具体的检测结果
    '''
    # 将所有数据转至指定设备
    img, (x_min, x_max, y_min, y_max) = img.to(device), (x_min.to(device), x_max.to(device), y_min.to(device), y_max.to(device))
    # 对图片进行检测，返回预测框[x_min,x_max,y_min,y_max]
    pred_position =net(img).squeeze(dim=2).squeeze(dim=2)
    # 对数据的后处理，以及坐标系的转换
    label_position = torch.stack([x_min, x_max, y_min, y_max], dim=0).T
    example_img = img[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
    example_img = example_img.copy()
    h = example_img.shape[0]
    example_pred_position = pred_position[0, :].cpu().detach().numpy()
    example_label_position = label_position[0, :].cpu().detach().numpy()
    pred_xmin = int(example_pred_position[0])
    pred_xmam = int(example_pred_position[1])
    pred_ymax = h - int(example_pred_position[2])
    pred_ymin = h - int(example_pred_position[3])
    label_xmin = int(example_label_position[0])
    label_xmax = int(example_label_position[1])
    label_ymax = h - int(example_label_position[2])
    label_ymin = h - int(example_label_position[3])
    # 返回统一坐标系后的检测结果
    return example_img,pred_xmin,pred_xmam,pred_ymin,pred_ymax,label_xmin,label_xmax,label_ymin,label_ymax

# 计算预测框和label框的IOU
def calculate_iou(pred_xmin,pred_xmam,pred_ymin,pred_ymax,label_xmin,label_xmax,label_ymin,label_ymax):

    # 获取两个框交叠部分的矩形的4个顶点
    inter_rect_x1 = torch.tensor(max(pred_xmin, label_xmin))
    inter_rect_y1 = torch.tensor(max(pred_ymin, label_ymin))
    inter_rect_x2 = torch.tensor(min(pred_xmam, label_xmax))
    inter_rect_y2 = torch.tensor(min(pred_ymax, label_ymax))

    # 计算交叠部分的矩形面积
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # 计算两个框各自的面积
    b1_area = (pred_xmam-pred_xmin) * (pred_ymax-pred_ymin)
    b2_area = (label_xmax-label_xmin) * (label_ymax-label_ymin)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

# 显示图片，并用白线画出预测框，黄线画出label框
def draw_pred_label(example_img,pred_xmin,pred_xmam,pred_ymin,pred_ymax,label_xmin,label_xmax,label_ymin,label_ymax):
    cv2.rectangle(example_img, (pred_xmin, pred_ymin), (pred_xmam, pred_ymax), (255, 255, 255), 2)
    cv2.rectangle(example_img, (label_xmin, label_ymin), (label_xmax, label_ymax), (0, 255, 255), 2)
    cv2.imshow("img", example_img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()