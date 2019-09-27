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
from models import get_model,Loss
import GPUtil as GPU

#get cuda here
def get_cuda_devices():
    device_list=[]
    device=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    device_list=[0,1]
    # if device.type=="cuda":
    #     device_list=GPU.getGPUs()
    return device,device_list

img_size=(499,499)


# augmentation_function = A.Compose([
#     A.CenterCrop(499,440,p=1),
#     A.Resize(512,512,p=1),
#     A.CLAHE(p=1),
# ], p=1)


augmentation_function = A.Compose([
    A.CenterCrop(499,440,p=1),
    A.Resize(512,512,p=1),
    A.CLAHE(p=1),
    A.OneOf([
	A.RandomContrast(),
    	A.RandomGamma(),
    	A.RandomBrightness(),
    	], p=0.3),
    A.OneOf([
    	A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    	A.GridDistortion(),
    	A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    	], p=0.3),
    A.ShiftScaleRotate(),
    ])



#transformer pipeline

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_n_classes(root: Path):
    return len(os.listdir(root))

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
            # print('going for augmentation')
            img = self.augment(img)

        #return img, self.image_path[idx],self.image_cat[idx],self.image_label[idx]
        return img_transform(img), self.image_path[idx],self.image_cat[idx],torch.tensor(int(self.image_label[idx]),dtype=torch.int64)


def validation(model: nn.Module, criterion, valid_loader):
    model.eval()
    losses = []
    total=0
    correct=0
    for i, (inputs,_,_, targets) in enumerate(valid_loader):

        inputs=inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        targets=targets.to(device)-1
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        loss = criterion(outputs, targets)
        batch_size = inputs.size(0)
        losses.append(loss.item())

    valid_loss = np.mean(losses)  # type: float
    accuracy=100*correct/total
    print('Valid loss: {:.5f},Accuracy : {:.5f}'.format(valid_loss,accuracy))
    metrics = {'valid_loss': valid_loss,'accuracy':accuracy}
    return metrics


def test(model: nn.Module, criterion, test_loader):
    model.eval()
    losses = []
    total=0
    correct=0
    for i, (inputs,_,_, targets) in enumerate(test_loader):

        inputs=inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        targets=targets.to(device)-1
        total += targets.size(0)
        correct += (preds == targets).sum().item()
        loss = criterion(outputs, targets)
        batch_size = inputs.size(0)
        losses.append(loss.item())

    test_loss = np.mean(losses)  # type: float
    accuracy=100*correct/total
    print('Test Loss loss: {:.5f},Accuracy : {:.5f}'.format(test_loss,accuracy))
    metrics = {'test_loss': test_loss,'accuracy':accuracy}
    return metrics

device,device_list=get_cuda_devices()
def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--size', type=str, default='512X512', help='Input size, for example 512X512. Must be multiples of 2')
    arg('--num_workers', type=int, default=4, help='Enter the number of workers')
    arg('--batch_size', type=int, default=16, help='Enter batch size')
    arg('--n_epochs', type=int, default=52, help='Enter number of epochs to run training for')
    arg('--report_each', type=int, default=10, help='Enter the span of last readings of running loss to report')
    arg('--lr', type=int, default=0.0001, help='Enter learning rate')
    arg('--fold_no', type=int, default=0, help='Enter the fold no')
    arg('--to_augment', type=bool, default=False, help='Augmentation flag')
    args = parser.parse_args()


    local_data_path = Path('.').absolute()
    local_data_path.mkdir(exist_ok=True)
    #mention the fold path here
    train_path=local_data_path/'..'/'input'/'train'
    a=CoinDataset(train_path,to_augment=args.to_augment)
    n_classes=get_n_classes(train_path)
    print(n_classes)
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
    #mention the fold path here
    train_path=local_data_path/'..'/'input'/'train'
    train_loader=make_loader(train_path,to_augment=args.to_augment, shuffle=True)
    validation_path=local_data_path/'..'/'input'/'validation'
    validation_loader=make_loader(validation_path,to_augment=args.to_augment, shuffle=True)
    test_path=local_data_path/'..'/'input'/'test'
    test_loader=make_loader(test_path,to_augment=args.to_augment, shuffle=True)

    #define model, and handle gpus

    print('device is',device)
    model_name='resnet18'
    model=get_model(model_name=model_name,pretrained_status=True,n_classes=n_classes).to(device)
    if device.type=="cuda":
        #model = nn.DataParallel(model, device_ids=device_list)
        print('cuda devices',device_list)

    #define optimizer and learning_rate
    init_optimizer=lambda lr: Adam(model.parameters(), lr=lr)
    lr=args.lr
    optimizer=init_optimizer(lr)
    criterion=Loss()
    #print(model)

    report_each=args.report_each
    #model save implementation
    model_path= local_data_path/'model_checkpoints'
    model_path.mkdir(exist_ok=True)
    model_path=local_data_path/'model_checkpoints'/'{model_name}_{fold}.pt'.format(model_name=model_name,fold=args.fold_no)
    best_model_path= local_data_path/'best_model_checkpoints'
    best_model_path.mkdir(exist_ok=True)
    best_model_path=local_data_path/'best_model_checkpoints'/'{model_name}_{fold}.pt'.format(model_name=model_name,fold=args.fold_no)
    #updated fold checkpoint here
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'best_valid_loss': best_valid_loss
    }, str(model_path))


    best_valid_loss = float('inf')
    valid_losses = []
    test_losses=[]
    valid_accuracy = []
    test_accuracy=[]
    for epoch in range(0, args.n_epochs):

        model.train()
        tq = tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        for i, (inputs,_,_, targets) in enumerate(train_loader):
            inputs=inputs.to(device)
            outputs = model(inputs)
            #start here
            _, preds = torch.max(outputs, 1)
            #end here
            targets=targets.to(device)-1
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            batch_size = inputs.size(0)
            tq.update(batch_size)
            losses.append(loss.item())
            mean_loss = np.mean(losses[-report_each:])
            tq.set_postfix(loss='{:.5f}'.format(mean_loss))
            (batch_size * loss).backward()
            optimizer.step()
            break
        tq.close()
        save(epoch)
        valid_metrics = validation(model, criterion, validation_loader)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        test_metrics = test(model, criterion, test_loader)
        test_loss = test_metrics['test_loss']
        test_losses.append(test_loss)
        if valid_loss < best_valid_loss:
            print('found better val loss model')
            best_valid_loss = valid_loss
            shutil.copy(str(model_path), str(best_model_path))


main()
