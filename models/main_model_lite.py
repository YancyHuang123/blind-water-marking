from asyncio import Task
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import lightning as L
import torch.nn as nn
from models.discriminator_satellite import DiscriminatorNet
from models.encoder import UnetGenerator
from models.resnet_satellite import Resnet18, Resnet34
from models.vgg import VGG
from torch.optim import SGD, Adam
import torch.nn as nn
import torch.nn.functional as F
from .loss import SSIM
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch
from utils.loss import SSIM
from torchmetrics.classification import MulticlassAccuracy
from Wrapper.WrapperModule import WrapperModule
# define the LightningModule


class MainModel(WrapperModule):  # type: ignore
    def __init__(self, logo):
        super().__init__()
        self.automatic_optimization = False

        self.encoder = UnetGenerator()
        self.discriminator = DiscriminatorNet()
        self.host_net = Resnet34()
        self.logo = logo

        self.host_net_no_trigger_acc = MulticlassAccuracy(num_classes=6)
        self.host_net_trigged_rate = MulticlassAccuracy(num_classes=6)
        self.host_net_error_trigged_rate = MulticlassAccuracy(num_classes=6)
        self.discriminator_acc = MulticlassAccuracy(num_classes=2)
        self.hyper_parameters = [3, 5, 1, 0.1]

    def configure_losses(self):
        encoder_mse_loss = nn.MSELoss()
        encoder_SSIM_loss = SSIM()
        host_net_loss = nn.CrossEntropyLoss()
        discriminator_loss = nn.BCELoss()
        return encoder_mse_loss, encoder_SSIM_loss, host_net_loss, discriminator_loss

    def training_step(self, batch, batch_idx):

        opt_encoder, opt_discriminator, opt_host_net = self.configure_optimizers()  # type: ignore
        encoder_mse_loss, encoder_SSIM_loss, host_net_loss, discriminator_loss = self.configure_losses()
        # training_step defines the train loop.
        # it is independent of forward
        (X, Y), (X_trigger, _) = batch
        trigger_size = X_trigger.shape[0]

        wm_labels = torch.full((X_trigger.shape[0], 1), 1).to(self.device)
        wm_labels = torch.squeeze(wm_labels)
        valid = torch.ones((trigger_size, 1), device=self.device)
        fake = torch.zeros((trigger_size, 1), device=self.device)
        logo_batch = self.logo.repeat(
            X_trigger.shape[0], 1, 1, 1).to(self.device)

        # train discriminator net
        opt_discriminator.zero_grad()  # type: ignore
        wm_img = self.encoder(X_trigger, logo_batch)
        wm_dis_output = self.discriminator(wm_img.detach())
        real_dis_output = self.discriminator(X_trigger)

        loss_D_wm = discriminator_loss(wm_dis_output, fake)
        loss_D_real = discriminator_loss(real_dis_output, valid)
        loss_D = loss_D_wm + loss_D_real
        self.manual_backward(loss_D)
        opt_discriminator.step()

        # train encoder
        opt_encoder.zero_grad()  # type: ignore
        opt_discriminator.zero_grad()  # type: ignore
        opt_host_net.zero_grad()  # type: ignore

        wm_dis_output = self.discriminator(wm_img)
        wm_dnn_output = self.host_net(wm_img)
        loss_mse = encoder_mse_loss(X_trigger, wm_img)
        loss_ssim = encoder_SSIM_loss(X_trigger, wm_img)
        loss_adv = discriminator_loss(wm_dis_output, valid)

        hyper_parameters = self.hyper_parameters

        loss_dnn = host_net_loss(wm_dnn_output, wm_labels)
        loss_H = (
            hyper_parameters[0] * loss_mse
            + hyper_parameters[1] * (1 - loss_ssim)
            + hyper_parameters[2] * loss_adv
            + hyper_parameters[3] * loss_dnn
        )
        self.manual_backward(loss_H)
        opt_encoder.step()

        # train host net
        opt_host_net.zero_grad()  # type: ignore

        inputs = torch.cat([X, wm_img.detach()], dim=0)  # type: ignore

        labels = torch.cat([Y, wm_labels], dim=0)
        dnn_output = self.host_net(inputs)

        loss_DNN = host_net_loss(dnn_output, labels)
        self.manual_backward(loss_DNN)
        opt_host_net.step()

        self.log_dict({"loss_D": loss_D, 'loss_H': loss_H, 'loss_mse': loss_mse, 'loss_ssim': loss_ssim,
                      'loss_adv': loss_adv, 'loss_DNN': loss_DNN}, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        encoder_mse_loss, encoder_SSIM_loss, host_net_loss, discriminator_loss = self.configure_losses()
        (X, Y), (X_trigger, Y_trigger) = batch

        wm_labels = torch.full((X_trigger.shape[0],), 1).to(self.device)
        valid = torch.FloatTensor(
            X_trigger.shape[0], 1).fill_(1.0).to(self.device)
        fake = torch.FloatTensor(
            X_trigger.shape[0], 1).fill_(0.0).to(self.device)
        fake = torch.squeeze(fake)
        valid = torch.squeeze(valid)

        logo_batch = self.logo.repeat(
            X_trigger.shape[0], 1, 1, 1).to(self.device)

        # evaluate discriminator net
        wm_img = self.encoder(X_trigger, logo_batch)

        wm_dis_output = self.discriminator(wm_img.detach())
        real_dis_output = self.discriminator(X_trigger)

        wm_dis_output = torch.squeeze(wm_dis_output)
        real_dis_output = torch.squeeze(real_dis_output)

        loss_D_wm = discriminator_loss(wm_dis_output, fake)
        loss_D_real = discriminator_loss(real_dis_output, valid)
        loss_D = loss_D_wm + loss_D_real

        self.discriminator_acc(
            torch.cat([wm_dis_output, real_dis_output], dim=0),
            torch.cat([fake, valid], dim=0),
        )

        # evaluate encoder net
        wm_dis_output = self.discriminator(wm_img)
        wm_dnn_output = self.host_net(wm_img)
        wm_dis_output = torch.squeeze(wm_dis_output)
        wm_dnn_output = torch.squeeze(wm_dnn_output)

        loss_mse = encoder_mse_loss(X_trigger, wm_img)
        loss_ssim = encoder_SSIM_loss(X_trigger, wm_img)
        loss_adv = discriminator_loss(wm_dis_output, valid)
        loss_dnn = host_net_loss(wm_dnn_output, wm_labels)

        hyper_parameters = self.hyper_parameters

        loss_H = (
            hyper_parameters[0] * loss_mse
            + hyper_parameters[1] * (1 - loss_ssim)
            + hyper_parameters[2] * loss_adv
            + hyper_parameters[3] * loss_dnn
        )

        # evaluate host net
        inputs = torch.cat([X, wm_img.detach()], dim=0)  # type: ignore

        original_labels = torch.cat([Y, Y_trigger], dim=0)

        dnn_output = self.host_net(inputs)
        dnn_trigger_pred = self.host_net(wm_img)

        error_trigger_labels = torch.where(original_labels == 1, 2, 1)

        loss_DNN = host_net_loss(dnn_output, original_labels)

        # update accuracies
        self.host_net_no_trigger_acc(dnn_output, original_labels)
        self.host_net_trigged_rate(dnn_trigger_pred, wm_labels)
        self.host_net_error_trigged_rate(dnn_output, error_trigger_labels)

        # log losses
        self.log_dict({"loss_D_val": loss_D, 'loss_H_val': loss_H, 'loss_mse_val': loss_mse, 'loss_ssim_val': loss_ssim,
                       'loss_adv_val': loss_adv, 'loss_DNN_val': loss_DNN}, on_step=False, on_epoch=True)
        self.log_dict({'host_net_no_trigger_acc': self.host_net_no_trigger_acc.compute(), 'host_net_trigged_rate':
                      self.host_net_trigged_rate.compute(), 'host_net_error_trigged_rate': self.host_net_error_trigged_rate.compute(), 'discriminator_acc': self.discriminator_acc.compute()}, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.host_net_error_trigged_rate.reset()
        self.host_net_no_trigger_acc.reset()
        self.host_net_trigged_rate.reset()
        self.discriminator_acc.reset()

    def configure_optimizers(self):
        opt_encoder = Adam(self.encoder.parameters(),
                           lr=0.001, betas=(0.5, 0.999))
        opt_discriminator = Adam(
            self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
        opt_host_net = SGD(self.host_net.parameters(), lr=0.01,
                           momentum=0.9, weight_decay=5e-4)
        return opt_encoder, opt_discriminator, opt_host_net
