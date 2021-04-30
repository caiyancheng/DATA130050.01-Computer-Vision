import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
from torchvision import models
from torchsummary import summary
from net.block import CBR
from config import device

#REF: https://github.com/meetshah1995/pytorch-semseg/

#先定义segnet中的基本单元
class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = CBR(in_size, out_size, 3, 1, 1)
        self.conv2 = CBR(out_size, out_size, 3, 1, 1)
        #在最大值池化中记录index
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.shape
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices

class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = CBR(in_size, out_size, 3, 1, 1)
        self.conv2 = CBR(out_size, out_size, 3, 1, 1)
        self.conv3 = CBR(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.shape
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.conv1 = CBR(in_size, in_size, 3, 1, 1)
        self.conv2 = CBR(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices):
        #在unpool的过程中传入indice索引
        outputs = F.max_unpool2d(inputs, indices, kernel_size=2, stride=2)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.conv1 = CBR(in_size, in_size, 3, 1, 1)
        self.conv2 = CBR(in_size, in_size, 3, 1, 1)
        self.conv3 = CBR(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices):
        outputs = F.max_unpool2d(inputs, indices, kernel_size=2, stride=2)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

#segnet网络
class SEGNET(nn.Module):
    def __init__(self, n_classes=2, in_channels=3, is_unpooling=True):
        super().__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        
        #与vgg中的conv大小基本一致
        #使用两个down2，三个down3
        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

    def forward(self, inputs):
        #在下采样时记录最大值池化的index和池化后的大小
        down1, indices_1 = self.down1(inputs)
        down2, indices_2 = self.down2(down1)
        down3, indices_3 = self.down3(down2)
        down4, indices_4 = self.down4(down3)
        down5, indices_5 = self.down5(down4)

        up5 = self.up5(down5, indices_5)
        up4 = self.up4(up5, indices_4)
        up3 = self.up3(up4, indices_3)
        up2 = self.up2(up3, indices_2)
        up1 = self.up1(up2, indices_1)

        return up1

    #同样用vgg16的参数初始化
    def init_vgg16_params(self):
        vgg16 = models.vgg16(pretrained=True).to(device)
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            #区分down2还是down3
            if idx < 2:
                units = [conv_block.conv1.cbr,conv_block.conv2.cbr]
            else:
                units = [conv_block.conv1.cbr,conv_block.conv2.cbr,conv_block.conv3.cbr,]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)
        
        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

if __name__ == '__main__':
    model = SEGNET().to(device)
    summary(model,(3,256,512))