import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

# logger = logging.getLogger(__name__)


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
#             transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),  transforms.ToTensor()
#             transforms.RandomCrop(size=224,
#                                   padding=int(224*0.125),
#                                   padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
#             transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=224,
#                                   padding=int(224*0.125),
#                                   padding_mode='reflect'),
            RandAugmentMC(n=2, m=10), transforms.ToTensor()
            ])
#         self.normalize = transforms.Compose([
#             transforms.ToTensor(),            
#             transforms.Normalize(mean=mean, std=std)])
    
        self.clean_transforms = transforms.Compose([
            
            transforms.ToTensor(),
            transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
#             transforms.RandomResizedCrop(224,
#                                          scale=(1.00, 1.2),
#                                          ratio=(0.75, 1.3333333333333333)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x = self.clean_transforms(x)
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong, x