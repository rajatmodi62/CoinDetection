import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.backends.cudnn
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from glob import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from torchvision.transforms import ToTensor, Normalize, Compose
from augmentation_pipeline import augment_and_show
img_size=(499,499)


augmentation_function = A.Compose([
    A.CenterCrop(499,440,p=1),
    A.Resize(512,512,p=1),
    A.CLAHE(p=1),
], p=1)

#transformer pipeline

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CoinDataset(Dataset):

    def augment(self,img):
        #augmentation_pipeline
        img=augment_and_show(augmentation_function, img)
        return img

    def convert_labels_to_names(self,path,label):
        with open(path) as json_file:
            data = json.load(json_file)
        return data[str(label)]

    def load_image(self,path: Path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,img_size,interpolation=cv2.INTER_AREA)
        return img.astype(np.uint8)

    def __init__(self, root: Path,labels_json='../input/cat_to_name.json' ,to_augment=False):
        # TODO This potentially may lead to bug.
        #self.image_paths = sorted(root.joinpath(mode).glob('/*/*.jpg'))
        self.image_path=[]
        self.image_cat=[]
        self.image_label=[]
        #fill the image_paths
        for image_folder in os.listdir(root):

            for image in os.listdir(root/image_folder):
                self.image_path.append(str(root/image_folder/image))
                self.image_cat.append(self.convert_labels_to_names(labels_json,int(image_folder)))
                self.image_label.append(image_folder)
        self.to_augment = to_augment

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = self.load_image(self.image_path[idx])

        if self.to_augment:
            #print('going for augmentation')
            img = self.augment(img)

        #return img, self.image_path[idx],self.image_cat[idx],self.image_label[idx]
        return img_transform(img), self.image_path[idx],self.image_cat[idx],self.image_label[idx]



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--size', type=str, default='512X512', help='Input size, for example 512X512. Must be multiples of 2')
    arg('--num_workers', type=int, default=4, help='Enter the number of workers')
    arg('--batch_size', type=int, default=16, help='Enter batch size')
    args = parser.parse_args()


    local_data_path = Path('.').absolute()
    local_data_path.mkdir(exist_ok=True)
    train_path=local_data_path/'..'/'input'/'train'
    a=CoinDataset(train_path,to_augment=True)

    '''

    num_workers,batch_size
    '''
    def make_loader(ds_root: Path, to_augment=False, shuffle=False):
        return DataLoader(
            dataset=CoinDataset(ds_root, to_augment=to_augment),
            shuffle=shuffle,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            pin_memory=True
        )

    #craeting a dataloader
    train_path=local_data_path/'..'/'input'/'train'
    train_loader=make_loader(train_path,to_augment=True, shuffle=True)
    validation_path=local_data_path/'..'/'input'/'validation'
    validation_loader=make_loader(validation_path,to_augment=True, shuffle=True)

    for i, (inputs, _,_,targets) in enumerate(train_loader):
        print(inputs.size())
main()