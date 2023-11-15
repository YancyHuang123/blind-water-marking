import torch
import torch.nn as nn
import models.discriminator_satellite as discriminator
from models.encoder import UnetGenerator
from models.resnet_satellite import Resnet18,Resnet34
from models.vgg import VGG
from torch.optim import SGD, Adam
import torch.nn as nn
from . import loss as L
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

class MainModel(nn.Module):
    def __init__(self, device='cuda',mutiGPU=True):
        super(MainModel, self).__init__()
        if mutiGPU == True:
            self.encoder = nn.DataParallel(UnetGenerator()).cuda()
            self.host_net = nn.DataParallel(Resnet34()).cuda()
            self.discriminator = nn.DataParallel(discriminator.DiscriminatorNet()).cuda()
        else:
            self.encoder = UnetGenerator().to(device)
            self.host_net = Resnet34().to(device)
            self.discriminator = discriminator.DiscriminatorNet().to(device)

        self.opt_encoder = Adam(self.encoder.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.opt_discriminator = Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.opt_host_net = SGD(self.host_net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        
        self.encoder_scheduler=ReduceLROnPlateau(self.opt_encoder,
                                                mode='min', factor=0.2, patience=5, verbose=True)
        
        self.discriminator_scheduler=ReduceLROnPlateau(self.opt_discriminator, 
                                                mode='min', factor=0.2, patience=8, verbose=True)
        
        self.host_net_scheduler=MultiStepLR(self.opt_host_net, milestones=[40, 80], gamma=0.1)
        
        
        self.encoder_mse_loss=nn.MSELoss()
        self.encoder_SSIM_loss=L.SSIM()
        self.host_net_loss=nn.CrossEntropyLoss()
        self.discriminator_loss=nn.BCELoss()
    
    def save_model(self,path):
        torch.save(self.state_dict(),f'{path}/main_model.pt')
        
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))