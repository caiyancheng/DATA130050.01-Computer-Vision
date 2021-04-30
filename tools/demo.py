import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv1',)
parse.add_argument('--weight-path', type=str, default='/remote-home/source/42/cyc19307140030/BisenetV1_new/tools/res/model_final_v1.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./picts/lindau_000014_000019_leftImg8bit.png',)
args = parse.parse_args()
cfg = cfg_factory[args.model]


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

print(args.weight_path)
# define model
net = model_factory[cfg.model_type](cfg.num_classes)#19
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
print(args.img_path)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# inference
out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
pred = palette[out]
cv2.imwrite('./picts/res/city/19class/lindau_000014_000019_leftImg8bit.jpg', pred)
