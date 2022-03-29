
import torch.nn as nn

# 检测模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(3,12,3),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(12, 128, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(128,32,3),
            nn.LeakyReLU()
        )
        self.head=nn.Sequential(
            nn.Conv2d(32,4,(23,32)),
            nn.LeakyReLU()
        )

    def forward(self,x):
        x=self.layer(x)
        return self.head(x)