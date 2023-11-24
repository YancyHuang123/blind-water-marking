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
from torch.utils.data import Dataset, DataLoader

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def get_dataset():
    # load trainset and testset
    trainset = torchvision.datasets.CIFAR10(
        root="./datas", train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./datas", train=False, download=True, transform=transform_test
    )

    trigger_train = torchvision.datasets.CIFAR10(
        root="./datas", train=True, download=True, transform=transform_test
    )

    trigger_test = torchvision.datasets.CIFAR10(
        root="./datas", train=False, download=True, transform=transform_test
    )

    return trainset, testset, trigger_train, trigger_test


def get_watermark():
    
    ieee_logo = torchvision.datasets.ImageFolder(
    root='./datas/logo', transform=transform_test)
    ieee_loader = DataLoader(ieee_logo, batch_size=1)
    it=iter(ieee_loader)
    logo=next(it)
    return logo[0][0]