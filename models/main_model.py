import torch
import torch.nn as nn
from models.discriminator import DiscriminatorNet
from models.encoder import UnetGenerator
from models.resnet import ResNet18
from models.vgg import VGG
from torch.optim import SGD, Adam
import torch.nn as nn


class MainModel(nn.Module):
    def __init__(self, mutiGPU=True):
        super(MainModel, self).__init__()
        if mutiGPU == True:
            self.encoder = nn.DataParallel(UnetGenerator()).cuda()
            self.resnet = nn.DataParallel(ResNet18()).cuda()
            self.discriminator = nn.DataParallel(DiscriminatorNet()).cuda()
        else:
            self.encoder = UnetGenerator().cuda()
            self.resnet = VGG('VGG19').cuda()
            self.discriminator = DiscriminatorNet().cuda()
        
        self.optimizer_encoder = Adam(self.encoder.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.optimizer_resnet = SGD(self.resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        

    def get_encoder(self):
        return self.encoder, self.optimizer_encoder

    def get_resnet(self):
        return self.resnet, self.optimizer_resnet

    def get_discriminator(self):
        return self.discriminator, self.optimizer_discriminator
