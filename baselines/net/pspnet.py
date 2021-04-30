import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import torchvision
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from config import device

#REF: https://github.com/mehtanihar/pspnet/blob/master/models/network.py

#psp中使用到的池化模块定义
class PyramidPool(nn.Module):

	def __init__(self, in_features, out_features, pool_size):
		super().__init__()

        #使用自适应池化变为给定pool_size的大小，再使用双线性插值上采样为原尺寸
		self.features = nn.Sequential(
			nn.AdaptiveAvgPool2d(pool_size),
			nn.Conv2d(in_features, out_features, 1, bias=False),
			nn.BatchNorm2d(out_features, momentum=0.95),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		size=x.size()
		output=F.interpolate(self.features(x), size[2:], mode='bilinear', align_corners=True)
		return output


class PSPNET(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        #使用resnet作为预训练的骨干网络，使用layer1-4提取特征，大小为原图大小的1/8
        self.resnet = torchvision.models.resnet50(pretrained = True)

        #layer5使用金字塔池化并拼接
        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)

        self.final = nn.Sequential(
        	nn.Conv2d(4096, 512, 3, padding=1, bias=False),
        	nn.BatchNorm2d(512, momentum=0.95),
        	nn.ReLU(inplace=True),
        	nn.Dropout(0.1),
        	nn.Conv2d(512, num_classes, 1),
        )


    def forward(self, x):
    
        size=x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        #得到每个子layer具有512个通道
        f5a = self.layer5a(x)
        f5b = self.layer5b(x)
        f5c = self.layer5c(x)
        f5d = self.layer5d(x) 

        #在通道上拼接
        y = torch.cat([x,f5a,f5b,f5c,f5d], 1)
        x = self.final(y)

        #直接通过双线性插值恢复为原大小
        x = F.interpolate(x,size[2:],mode='bilinear',align_corners=True)
       
        return x

if __name__ == '__main__':
    model = PSPNET().to(device)
    summary(model,(3,256,512))
