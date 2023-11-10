import torch
import torch.nn as nn
import models.discriminator_satellite as discriminator
from models.encoder import UnetGenerator
from models.resnet_satellite import Resnet34
from models.vgg import VGG
from torch.optim import SGD, Adam
import torch.nn as nn
from . import loss as L

class MainModel(nn.Module):
    def __init__(self, device,mutiGPU=True):
        super(MainModel, self).__init__()
        if mutiGPU == True:
            self.encoder = nn.DataParallel(UnetGenerator()).cuda()
            self.host_net = nn.DataParallel(Resnet34()).cuda()
            self.discriminator = nn.DataParallel(discriminator.DiscriminatorNet()).cuda()
        else:
            self.encoder = UnetGenerator().to(device)
            self.host_net = Resnet34().to(device)
            self.discriminator = discriminator.DiscriminatorNet().to(device)

        self.opt_encoder = Adam(self.encoder.parameters(), lr=0.001)
        self.opt_discriminator = Adam(self.discriminator.parameters(), lr=0.001)
        self.opt_host_net = Adam(self.discriminator.parameters(), lr=0.001)
        
        self.encoder_mse_loss=nn.MSELoss()
        self.encoder_SSIM_loss=L.SSIM()
        self.host_net_loss=nn.CrossEntropyLoss()
        self.discriminator_loss=nn.BCELoss()
    
    def save(self,path):
        torch.save(self.state_dict(),path)