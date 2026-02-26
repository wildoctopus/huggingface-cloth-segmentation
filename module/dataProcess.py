import torch
from torch.utils import data

from pathlib import Path

import torch
# import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"
            

class ImgSet(data.Dataset):

    def __init__(self, imgFolder:Path):
        self.imgPaths=[img for img in imgFolder.glob('*.jpg')]
        # print(self.imgList)

        transfm_list=[transforms.ToTensor(),
                      Normalize_image(0.5, 0.5),
                      ]
        self.transfm=transforms.Compose(transfm_list)

    def __getitem__(self, index):
        imgPath=self.imgPaths[index]
        imgName=imgPath.stem
        # print(imgName)
        img=Image.open(str(imgPath))
        img_size = img.size
        img = img.resize((768, 768), Image.BICUBIC)

        imgTensor=self.transfm(img)
        imgTensor = torch.unsqueeze(imgTensor, 0)#[1,3,H,W]
        return imgName, img_size, imgTensor
    
    def __len__(self):
        return len(self.imgPaths)
    
    def outputDir(self):
        dirs=[]
        folders=['fin_seg', 'Upper_body', 'Lower body', 'Full body']
        for i, name in enumerate(folders):
            f=Path('output')
            if i!=0:
                f=f/'alpha'
            f=f/name
            f.mkdir(parents=True,exist_ok=True)
            dirs.append(f)
        return dirs

            

