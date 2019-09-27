import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
from torchvision import datasets, models, transforms

model_list= ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



def get_model(model_name='resnet34',pretrained_status=False,n_classes=2):
    if model_name not in model_list:
        print('model not found')
        return -1;
    if model_name=='resnet18':
        model = models.resnet18(pretrained=pretrained_status)
    if model_name=='resnet34':
        model = models.resnet34(pretrained=pretrained_status)
    if model_name=='resnet50':
        model = models.resnet50(pretrained=pretrained_status)
    if model_name=='resnet101':
        model = models.resnet101(pretrained=pretrained_status)
    if model_name=='resnet152':
        model = models.resnet152(pretrained=pretrained_status)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    return model

class Loss:
    def __init__(self):
        self.nll_loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        return loss
