import torchvision
import torchvision.transforms as transforms
from lightning.dataloader import MyLoader
import torchvision.io as io
import cv2
import torchvision.transforms as transforms
import torch

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_dataset():
    # load trainset and testset
    trainset = torchvision.datasets.CIFAR10(
        root='./datas', train=True, download=True, transform=transform_train)

    testset = torchvision.datasets.CIFAR10(
        root='./datas', train=False, download=True, transform=transform_test)
    return trainset, testset


def get_watermark():
    # load logo
    logo = cv2.imread("./datas/logo/new-ieeelogo.png")
    logo = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
    logo = transform_test(logo)
    logo = logo.clone().detach() # type: ignore

    return logo
