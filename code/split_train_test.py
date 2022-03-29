import os
import pickle
'''
    划分训练集和测试集的数据
'''
#文件夹目录
img_path = '../Capture'
# 获取当前路径下的文件名，返回List
fileNames = os.listdir(img_path)
# 存取作为训练集的图片的文件名
train_img=[]
# 存取作为训练集的label的文件名
train_label=[]
# 存取作为测试集的图片的文件名
test_img=[]
# 存取作为测试集的label的文件名
test_label=[]

#遍历文件夹
# 根据文件名里面的数字进行划分，最后训练集为431张图片，测试集为213张图片
for file in fileNames:
    img_folder = img_path + '/' + file
    print(img_folder)
    if os.path.exists(img_folder):
        for image_name in os.listdir(img_folder):
            # 判断是bmp还是label
            if image_name[-3:]=="bmp":
                if int(image_name[11:14])%30!=0:
                    train_img.append(img_folder+'/'+image_name)
                else:
                    test_img.append(img_folder+'/'+image_name)
            else:
                if int(image_name[11:14])%30!=0:
                    train_label.append(img_folder+'/'+image_name)
                else:
                    test_label.append(img_folder+'/'+image_name)

# 将所有图片、label分成训练集、测试集后的结果，用pickle文件保存起来，在训练和测试时先读取文件名列表，再读取具体的数据
with open('../data/train_img.pkl', 'wb') as f:
    pickle.dump(train_img, f)

with open('../data/test_img.pkl', 'wb') as f:
    pickle.dump(test_img, f)

with open('../data/train_label.pkl', 'wb') as f:
    pickle.dump(train_label, f)

with open('../data/test_label.pkl', 'wb') as f:
    pickle.dump(test_label, f)
