import json
import os
import torchvision
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

# from https://github.com/pytorch/vision/tree/master/torchvision
# REF: https://blog.csdn.net/weixin_41424926/article/details/105383064

class myCityscapes(VisionDataset):

    #Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            target_type: Union[List[str], str] = "instance",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        if mode == 'fine':
            self.mode = 'gtFine'
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        elif mode == 'person':
            self.mode = 'gtBbox'
        elif mode == 'foggy':
            self.mode = 'gtFine'
        #self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'

        #图片路径
        if mode == 'foggy':
            self.images_dir = os.path.join(self.root, 'leftImg8bit_foggy',split)
        else:
            self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        
        #标签路径
        if mode == 'person':
            self.targets_dir = os.path.join(self.root, 'gtBboxCityPersons', split)
        else:
            self.targets_dir = os.path.join(self.root, self.mode, split)

        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse","person",'foggy'))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color", "person"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))
            elif self.mode == 'gtBbox':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_cityPersons_trainval.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon' or t == 'person':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        elif target_type == 'polygon':
            return '{}_polygons.json'.format(mode)
        elif target_type == 'person':
            return 'gtBboxCityPersons.json'

IMG_SIZE = 1024
MAX_OB = 30
MIN_H = 20
MIN_W = 20

myTransform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def jsonTransform(x):
    obs = x['objects']
    
    #设置最多物体数
    boxes = torch.Tensor(np.zeros((MAX_OB,5)))
    cnt = 0
    for ob in obs:
        if  ob['label'] == 'ignore':
            continue
        else:
            boxes[cnt][4] = 1
            boxes[cnt][:4] = torch.Tensor(ob['bbox']) / IMG_SIZE
            cnt += 1
            if cnt == MAX_OB:
                break 
    return boxes

def np2rets(x,h,w):
    rets = []
    for xx in x:
        
        if xx[4]==0:
            break
        xx = IMG_SIZE * xx 
        #print(xx)
        #数据格式，bbox左上角点、宽、高,cls
        ret = patches.Rectangle((xx[0],xx[1]),xx[2],xx[3],linewidth=1, edgecolor='r',facecolor='none') 
        rets.append(ret)
    return rets 

def show_labels_img(img,label,savepath):
    img = img.float().detach().cpu().permute(1,2,0).numpy()
    h,w,_ = img.shape
    label = label.detach().cpu().numpy()
    ax =plt.subplot(1,1,1)
    plt.imshow(img)
    rets = np2rets(label,h,w)
    for ret in rets:
        ax.add_patch(ret)
    plt.axis('off')
    plt.savefig(savepath)


if __name__ == '__main__':
    BATCH = 4
    dataset = myCityscapes('./datasets/', split='train', mode='person', target_type='person',transform=myTransform,target_transform=jsonTransform)
    dataloader = DataLoader(dataset, batch_size=BATCH,shuffle=True)
    for i,data in enumerate(dataloader):
        if i > 0:
            break
        img,target = data
        img = img[0]
        target = target[0]
        show_labels_img(img,target,'dumping/pic.png')