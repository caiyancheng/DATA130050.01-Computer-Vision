import torch
import torch.nn as nn
from torchvision import models
import torchvision
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from config import device

#图像级联网络
#REF: https://github.com/liminn/ICNet-pytorch/

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#PSPNET中的金字塔池化，默认参数和PSPNET相同，为[1,2,3,6]
class PyramidPooling(nn.Module):
	def __init__(self, pyramids=[1,2,3,6]):
		super().__init__()
		self.pyramids = pyramids

	def forward(self, input):
		feat = input
		height, width = input.shape[2:]
		for bin_size in self.pyramids:
			x = F.adaptive_avg_pool2d(input, output_size=bin_size)
			x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)
			feat  = feat + x
		return feat

# CCF单元，级联上下层特征，将底层特征通过分类器输出辅助分类结果，并将低层与高层特征进行结合
class CCF(nn.Module):

    def __init__(self, low_channels, high_channels, out_channels, nclass):
        super().__init__()
        #使用dilation进行空洞卷积
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        #分类器
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        #将x_low插值经过上采样插值为x_high的大小
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls

class ICHead(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.cff_12 = CCF(128, 64, 128, nclass)
        self.cff_24 = CCF(2048, 512, 128, nclass)
        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        outputs.append(up_x8)
        # 输出不同大小的图片，1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()

        return outputs

class ICNET(nn.Module):
    #使用resnet作为预训练的模型
    def __init__(self, nclass = 2):
        super().__init__()
        self.backbone = models.resnet50(pretrained = True)
        #使用步长为2的卷积
        self.conv_sub1 = nn.Sequential(CBR(3, 32, 3, 2),CBR(32, 32, 3, 2),CBR(32, 64, 3, 2))
        self.ppm = PyramidPooling()
        self.head = ICHead(nclass)

    #输入图片x，使用backbone提取各层特征
    def getFeatures(self,x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        return c1, c2, c3, c4

    def forward(self, x):
        size = x.size()[2:]
        #原大小,提取特征为1/8 eg.[32,64]
        x_sub1 = self.conv_sub1(x)
        
        #1/2图的特征，提取特征为1/16 eg.[16,32]
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        _, x_sub2, _, _ = self.getFeatures(x_sub2)
        
        #1/4图，并加入图像金字塔池化
        x_sub4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        _, _, _, x_sub4 = self.getFeatures(x_sub4)
        x_sub4 = self.ppm(x_sub4)
        #通过IChead分类头输出结果

        outputs = self.head(x_sub1, x_sub2, x_sub4)
        
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=size, mode='bilinear', align_corners=True)
        
        return outputs

class ICLoss(nn.Module):
    def __init__(self,weights=[1.0,0.2,0.2,0]):
        super().__init__()
        self.critirion = nn.CrossEntropyLoss()
        self.weights =  weights

    def forward(self,outputs,y):
        loss = 0
        for out,w in zip(outputs,self.weights):
            loss += w * self.critirion(out, y)
        return loss 

if __name__ == '__main__':
    model = ICNET().to(device)
    summary(model,(3,256,512))