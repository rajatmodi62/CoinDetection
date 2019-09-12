# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:44:22 2019

@author: hp
"""

import torch
import torchvision
import PIL

from PIL import Image

    
    img = Image.open('C:/Users/hp/Desktop/Coin_dataset_31/train/1/004__1 Cent_australia.jpg')
    img = img.resize((499,499), PIL.Image.ANTIALIAS)
#img.show()

    transform_centercropped= torchvision.transforms.CenterCrop((499,450))

    centercropped_image = transform_centercropped(img)


    centercropped_image.show()
    centercropped_image.save('C:/Users/hp/Desktop/Coin_dataset_31/train/1/cropped'+'1.jpg')