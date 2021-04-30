import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import models
from torchvision import transforms
from torchvision import models
from PIL import Image
from torch.utils.data import DataLoader
from net.fcn import FCN32,FCN16,FCN8
from net.linknet import LINKNET
from net.segnet import SEGNET
from net.pspnet import PSPNET
from net.icnet import ICNET, ICLoss
from net.unet import UNET
import torch.nn as nn
import torch.nn.functional as F
import tqdm 
from torchvision.utils import make_grid
import time 
from torchsummary import summary
from data import myCityscapes
from config import device
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#设置太大的BATCH会爆显存
BATCH = 8

def labelTransform(x):
    x = transforms.Resize(256)(x)
    x = np.array(x)
    x = torch.LongTensor(x)
    #第7类为道路
    mask = torch.LongTensor([[7]]).expand_as(x)
    y = (x==mask).long()
    return y 

myTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#mode = 'foggy' or 'fine'
dataset = myCityscapes('./datasets/', split='train', mode='fine', target_type='semantic',transform=myTransform,target_transform=labelTransform)
dataloader = DataLoader(dataset, batch_size=BATCH,shuffle=True)
validset = myCityscapes('./datasets/', split='val', mode='fine', target_type='semantic',transform=myTransform,target_transform=labelTransform)
validloader = DataLoader(validset, batch_size=BATCH, shuffle=True)

#fcn = models.segmentation.fcn_resnet50(pretrained=True).to(device)

class SegModel():
    def __init__(self):
        #demo使用很少的训练轮次
        self.epoch = 10
        self.lr = 1e-4

        #FCN/SEGNET基于vgg16进行初始化
        self.model = FCN8().to(device)
        self.model.init_vgg16_params()
        #self.model = SEGNET().to(device)
        #self.model.init_vgg16_params()

        #使用残差卷积的LNINNET
        #self.model = LINKNET().to(device)

        #用resnet作为backbone并加入金字塔的PSPNET
        #self.model = PSPNET().to(device)

        #self.model = UNET().to(device)
        self.critirion = nn.CrossEntropyLoss()

        #self.model = ICNET().to(device)
        #self.critirion = ICLoss()

        self.optimizer =  torch.optim.Adam(self.model.parameters(),lr=self.lr)
    
    def train(self,dataloader,validloader):
        train_acces = []
        valid_acces = []
        losses = []
        for e in range(self.epoch):
            Loss = 0
            
            #共2975张图片
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                #demo用比较少图片的训练
                if i > 50:
                    break

                self.optimizer.zero_grad()
                x,y = data
                x = x.to(device)
                y = y.to(device)

                out = self.model(x)
               
                loss = self.critirion(out,y)
                loss.backward()
                self.optimizer.step()

                Loss += loss

            Loss /= len(dataloader)
            losses.append(Loss)
            print("epoch: ",e, "loss: ",Loss.item())
            train_acc,_,_ = self.getMetrics(dataloader)
            valid_acc,_,_ = self.getMetrics(validloader)
            train_acces.append(train_acc)
            valid_acces.append(valid_acc)

            savepath = "results/" + str(e) + ".png"
            self.getPicture(dataloader, savepath)

        plt.figure()
        plt.plot(train_acces)
        plt.plot(valid_acces)
        plt.legend(["train","valid"])
        plt.savefig("results/acc.png")

        plt.figure()
        plt.plot(losses)
        plt.savefig("results/loss.png")

    def test(self,dataloader):
        acc,miou,fps = self.getMetrics(dataloader)
        print("accuracy: ", acc, "MIou: ", miou, "fps: ",fps) 

    @torch.no_grad()
    def getMetrics(self,dataloader):
        correct = 0
        tot_pixel = 0
        tot_time = 0
        tot_num = 0

        tot_intersect = 0
        tot_union = 0
        
        for i,data in enumerate(dataloader):
            #选取部分计算评价指标
            if i > 5:
                break
            x,y = data
            B = x.shape[0]
            x = x.to(device)
            y = y.to(device)

            tot_num += 1

            start_time = time.time()
            #ICNET取第一个元素为输出
            out = self.model(x)
            if isinstance(out,list):
                out = out[0]
            end_time = time.time()
            tot_time += end_time - start_time

            pred = torch.argmax(out,dim=1)
            
            correct += torch.sum(pred ==  y)
            tot_pixel += torch.sum((y>=0))

            pred = pred.bool()
            y = y.bool()
            tot_intersect += torch.sum(pred & y) 
            tot_union += torch.sum(pred | y)

        acc = (correct / tot_pixel).item()
        miou = (tot_intersect / tot_union).item()
        fps = (tot_num*BATCH) / tot_time
    
        return acc,miou,fps
            

    def getPicture(self,dataloader,path):
        for i,data in enumerate(dataloader):
            if i>0:
                break
            x,y = data
            x = x.to(device)
            
            plt.figure() 
            plt.subplot(2,1,1)
            out = self.model(x)
            if isinstance(out,list):
                out = out[0]
            pred = torch.argmax(out,dim=1).float().detach().cpu()
            pred = pred.unsqueeze(dim=1)
            pic = make_grid(pred,padding=2).permute(1,2,0)
            plt.imshow(pic)
            plt.axis('off')

            plt.subplot(2,1,2)
            y = y.float().detach().cpu()
            y = y.unsqueeze(dim=1)
            pic = make_grid(y,padding=2).permute(1,2,0)
            plt.imshow(pic)
            plt.axis('off')
            plt.savefig(path)

seg = SegModel()
#summary(seg.model, (3,256,512))
seg.train(dataloader,validloader)
seg.test(validloader) #使用valloader而不使用testloader，猜测testdata标注有误
torch.save(seg.model.state_dict(), 'pth/segnet.pth')
