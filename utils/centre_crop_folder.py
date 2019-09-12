# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:44:22 2019

@author: hp
"""

import torch
import torchvision
import PIL
from glob import glob
import numpy as np

from PIL import Image

img_list = sorted(glob('../../dataset/train/*/*.jpg'))
print(len(img_list))
for i,img_path in enumerate(img_list):
    #img = Image.open('../../dataset/train/1/004__1 Cent_australia.jpg')
    img=Image.open(img_path)
    img = img.resize((499,499), PIL.Image.ANTIALIAS)
    #img.show()

    transform_centercropped= torchvision.transforms.CenterCrop((499,450))

    centercropped_image = transform_centercropped(img)
    #print(img_path)
    new_img_path=img_path[:-4]+'_centercropped.jpg'
    #print(new_img_path)
    print(i)
    # centercropped_image.show()
    centercropped_image.save(new_img_path)
