import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
from torchvision import models

#先定义辅助单元

#CBR/CR..etc
class CBR(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding)
        self.cbr = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=inplace))
    
    def forward(self,x):
        return self.cbr(x)

class CR(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding,inplace=True):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding)
        self.cr = nn.Sequential(conv, nn.ReLU(inplace=inplace))
    
    def forward(self,x):
        return self.cr(x)

class CB(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride,padding):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        conv = nn.Conv2d(in_channels,out_channels,k_size,stride,padding)
        self.cb = nn.Sequential(conv, nn.BatchNorm2d(out_channels))
    def forward(self,x):
        return self.cb(x)

class DCBR(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, padding,inplace=True):
        super().__init__()

        self.dcbr = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(out_channels),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=inplace),
        )

    def forward(self, inputs):
        outputs = self.dcbr(inputs)
        return outputs

