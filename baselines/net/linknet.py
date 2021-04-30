import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
from torchvision import models
from torchsummary import summary
from net.block import CBR, DCBR, CB
from config import device

#REF: https://github.com/meetshah1995/pytorch-semseg/

#残差卷积

#optimizer:RMSProp lr=5e-4
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.cbr = CBR(in_channels, out_channels, 3, stride, 1)
        self.cb = CB(out_channels, out_channels, 3, 1, 1)
        #在CB R之间引入残差单元
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        residual = x
        out = self.cbr(x)
        out = self.cb(out)
        out += residual
        out = self.relu(out)
        return out

#带下采样的残差卷积
class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.cbr = CBR(in_channels, out_channels, 3, stride, 1)
        self.cb = CB(out_channels, out_channels, 3, 1, 1)
        self.downsample = CBR(in_channels, out_channels,3, stride, 1)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.cbr(x)
        out = self.cb(out)

        out += residual
        out = self.relu(out)
        return out

#上采样，通过反卷积操作
class linknetUp(nn.Module):
    def __init__(self, in_channels, out_channels,inplace=True):
        super().__init__()
        
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.cbr1 = CBR(in_channels, out_channels/2, k_size=1, stride=1, padding=0, inplace=inplace)
        self.dcbr2 = DCBR(out_channels/2, out_channels/2, k_size=2, stride=2, padding=0, inplace=inplace)
        self.cbr3 = CBR(out_channels/2, out_channels, k_size=1, stride=1, padding=0, inplace=inplace)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.dcbr2(x)
        x = self.cbr3(x)
        return x

#LINKNET将编码器和解码器相连接
class LINKNET(nn.Module):
    def __init__(self, n_classes=2, in_channels=3):
        super().__init__()

        self.firstconv = CBR(in_channels, 16, 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #编码器
        self.encoder1 = nn.Sequential(ResBlock(16,16,1,False), ResBlock(16,16,1,False))
        self.encoder2 = nn.Sequential(ResDownBlock(16,32,2,False), ResBlock(32,32,1,False))
        self.encoder3 = nn.Sequential(ResDownBlock(32,64,2,False), ResBlock(64,64,1,False))
        self.encoder4 = nn.Sequential(ResDownBlock(64,128,2,False), ResBlock(128,128,1,False))

        #解码器，将编码器中的信息连接到解码器上

        #跳层连接的过程中，relu的inplace应该设置为False
        self.decoder4 = linknetUp(128, 64, inplace=False)
        self.decoder3 = linknetUp(64, 32, inplace=False)
        self.decoder2 = linknetUp(32, 16, inplace=False)
        self.decoder1 = linknetUp(16, 16, inplace=False)

        #分割产生器
        self.finaldeconv1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 2, 2, 0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.finalconv2 = CBR(8,8,3,1,1)
        self.finalconv3 = nn.Conv2d(8, 2, 3, 1, 1)

    def forward(self, x):
        #编码, encoder234逐次下采样
        x = self.firstconv(x)
        x = self.maxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
       
        #将四个编码器的结果加到解码器上
        d4 = self.decoder4(e4)
        
        d4 += e3
        d3 = self.decoder3(d4)
        d3 += e2
        d2 = self.decoder2(d3)
        d2 += e1
        d1 = self.decoder1(d2)

        #产生分割结果
        f1 = self.finaldeconv1(d1)
        f2 = self.finalconv2(f1)
        f3 = self.finalconv3(f2)

        return f3

if __name__ == '__main__':
    model = LINKNET().to(device)
    #with open('./dump/linknet','w') as f: 
    #   f.write(str(model))
    summary(model,(3,256,512))