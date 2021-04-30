import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
from torchvision import models
from config import device

#Unet: 在FCN的基础上，加上了高层与低层之间的连接

#REF: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py

class UNET(nn.Module):
    def __init__(self, feature_scale=4, n_classes=2, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # 下采样，默认使用BN
        self.conv1 = unetConv2(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        #UNet最中间一层
        self.center = unetConv2(filters[3], filters[4])

        # 上采样，默认采用含有可学习参数的反卷积
        self.up_concat4 = unetUp(filters[4], filters[3])
        self.up_concat3 = unetUp(filters[3], filters[2])
        self.up_concat2 = unetUp(filters[2], filters[1])
        self.up_concat1 = unetUp(filters[1], filters[0])

        #用卷积实现
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

#unet中的卷积单元，每个卷积单元由两个卷积组成，默认使用BN
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU())
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

#unet中的上采样模块，将通道提升为两倍
class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv=True):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))