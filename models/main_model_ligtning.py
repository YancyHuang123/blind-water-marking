from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import torch.nn as nn
from models.discriminator_satellite import DiscriminatorNet
from models.encoder import UnetGenerator
from models.resnet_satellite import Resnet18,Resnet34
from models.vgg import VGG
from torch.optim import SGD, Adam
import torch.nn as nn
from .loss import SSIM
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau


# define the LightningModule
class MainModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization=False

        self.encoder=UnetGenerator()
        self.discriminator=DiscriminatorNet()
        self.host_net=Resnet34()
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x,y = batch
        
        
        
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
