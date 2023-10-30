import torchvision
import torchvision.transforms as transforms
from lightning.dataloader import MyLoader
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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

seed=32

cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True


def get_dataset():
    # load trainset and testset
    trainset = torchvision.datasets.CIFAR10(
        root='./datas', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./datas', train=False, download=True, transform=transform_test)

    trigger_set = torchvision.datasets.CIFAR10(
        root='./datas', train=True, download=True, transform=transform_test)

    return trainset, testset, trigger_set


def get_watermark():
    # load logo
    logo = cv2.imread("./datas/logo/IEEE/new-ieeelogo.png")
    logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
    logo = transform_test(logo)
    logo = logo.clone().detach()  # type: ignore

    return logo
