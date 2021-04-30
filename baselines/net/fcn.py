import torch.nn as nn
import torch.nn.functional as F
import torch 
import torchvision
from torchvision import models
from torchsummary import summary
from config import device
# REF: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/fcn.py

class FCN32(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        #self.loss = functools.partial(cross_entropy2d, size_average=False)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=32)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        out = self.classifier(conv5)
        out = self.upsample(out)
        return out

    #复制vgg的卷积层参数
    def init_vgg16_params(self):
        vgg16 = models.vgg16(pretrained=True).to(device)
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        #复制vgg16的参数
        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

class FCN16(FCN32):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        #将conv4的信息连结
        score = F.interpolate(score, score_pool4.size()[2:],mode="bilinear",align_corners=True)
        score += score_pool4
        out = F.interpolate(score, x.size()[2:],mode="bilinear",align_corners=True)

        return out

class FCN8(FCN32):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        score = F.interpolate(score, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        score += score_pool4
        score = F.interpolate(score, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        score += score_pool3
        out = F.interpolate(score, x.size()[2:], mode='bilinear', align_corners=True)

        return out

if __name__ == '__main__':
    fcn8 = FCN8().to(device)
    fcn16 = FCN16().to(device)
    fcn32 = FCN32().to(device)

   
    summary(fcn32,(3,256,512))
    
    