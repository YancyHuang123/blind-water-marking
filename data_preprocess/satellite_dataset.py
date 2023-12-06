import torchvision
import torchvision.transforms as transforms
from utils.dataloader import MyLoader
import torchvision.io as io
import cv2
import torchvision.transforms as transforms
import torch
import numpy as np
import time
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
transform_test = transforms.Compose([
    transforms.ToTensor()])

def get_dataset():
    # load trainset and testset
    trainset = torchvision.datasets.ImageFolder(
        root='./datas/IntelImage/seg_train/seg_train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(
        root='./datas/IntelImage/seg_test/seg_test', transform=transform_test)
    
    trigger_train=torchvision.datasets.ImageFolder(
        root='./datas/IntelImage/seg_train/seg_train', transform=transform_train)
    trigger_test = torchvision.datasets.ImageFolder(
        root='./datas/IntelImage/seg_test/seg_test', transform=transform_test)

    return trainset, testset, trigger_train,trigger_test


def get_watermark(path):
    # load logo
    logo = cv2.imread(path)
    logo = transform_test(logo)
    
    return logo
