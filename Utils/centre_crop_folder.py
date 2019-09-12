# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:30:01 2019

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:44:22 2019

@author: hp
"""
import os
import torch
import torchvision
import PIL
import cv2
from PIL import Image


imgs = os.listdir('C:/Users/hp/Desktop/data/train/1')

for img_name in imgs:
    
    img = Image.open('C:/Users/hp/Desktop/Coin_dataset_31/train/1'+'//'+img_name)
    cv2.imread('C:/Users/hp/Desktop/Coin_dataset_31/train/1'+'//'+img_name)
    img = img.resize((499,499), PIL.Image.ANTIALIAS)
#img.show()

    transform_centercropped= torchvision.transforms.CenterCrop((499,450))

    centercropped_image = transform_centercropped(img)


    #centercropped_image.show()
    centercropped_image=centercropped_image.save(r"C:/Users/hp/Desktop/data/train/1/cropped/final_image.jpg")
    #cv2.imwrite('C:/Users/hp/Desktop/Coin_dataset_31/train/1/cropped/cropped_img', centercropped_image)