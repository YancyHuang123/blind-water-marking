from idna import valid_contextj
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
import torch
from utils.loss import SSIM

# define the LightningModule
class MainModel(L.LightningModule): # type: ignore
    def __init__(self,logo):
        super().__init__()
        self.automatic_optimization=False   

        self.encoder=UnetGenerator()
        self.discriminator=DiscriminatorNet()
        self.host_net=Resnet34()
        self.logo=logo

        self.encoder_mse_loss=nn.MSELoss()
        self.encoder_SSIM_loss=SSIM()
        self.host_net_loss=nn.CrossEntropyLoss()
        self.discriminator_loss=nn.BCELoss()
        
    def training_step(self, batch, batch_idx):
        opt_encoder,opt_discriminator,opt_host_net = self.optimizers() # type: ignore
        # training_step defines the train loop.
        # it is independent of forward
        (X,Y),(X_trigger,_) = batch
        batch_size=X.shape[0]
        trigger_size=X_trigger.shape[0]

        valid = torch.ones((trigger_size,1), device=self.device)
        fake = torch.zeros((trigger_size,1), device=self.device)
        logo_batch = self.logo.repeat(X_trigger.shape[0], 1, 1, 1).to(self.device)
        # train discriminator net
        opt_discriminator.zero_grad() # type: ignore
        wm_img=self.encoder(X_trigger,logo_batch)
        wm_dis_output = self.discriminator(wm_img.detach())
        real_dis_output = self.discriminator(X_trigger)

        loss_D_wm = self.discriminator_loss(wm_dis_output, fake)
        loss_D_real = self.discriminator_loss(real_dis_output, valid)
        loss_D = loss_D_wm + loss_D_real
        self.manual_backward(loss_D)
        opt_discriminator.step()

        #loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log_dict({"loss_D_wm": loss_D_wm, "loss_D":loss_D}, prog_bar=True)
        

    def configure_optimizers(self):
        opt_encoder = Adam(self.encoder.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_discriminator = Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_host_net = SGD(self.host_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        return opt_encoder,opt_discriminator,opt_host_net
