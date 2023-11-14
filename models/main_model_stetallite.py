from typing import Any
from numpy import number
import torch
import torch.nn as nn
from models.discriminator import DiscriminatorNet
from models.encoder import UnetGenerator
from models.resnet_satellite import ResNet34
from models.vgg import VGG
from torch.optim import SGD, Adam
import torch.nn as nn
from . import loss as L
import os
from datetime import datetime


class MainModel(nn.Module):
    def __init__(self, mutiGPU=True):
        super(MainModel, self).__init__()
        if mutiGPU == True:
            self.encoder = nn.DataParallel(UnetGenerator()).cuda()
            self.host_net = nn.DataParallel(ResNet34()).cuda()
            self.discriminator = nn.DataParallel(DiscriminatorNet()).cuda()
        else:
            self.encoder = UnetGenerator().cuda()
            self.host_net = ResNet34().cuda()
            self.discriminator = DiscriminatorNet().cuda()

        self.opt_encoder = Adam(self.encoder.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.opt_discriminator = Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.opt_host_net = SGD(self.host_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        self.encoder_mse_loss = nn.MSELoss()
        self.encoder_SSIM_loss = L.SSIM()
        self.host_net_loss = nn.CrossEntropyLoss()
        self.discriminator_loss = nn.BCELoss()

    def save_model(self, check_folder):
        torch.save(self.state_dict(), f'{check_folder}/main_model.pt')

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
