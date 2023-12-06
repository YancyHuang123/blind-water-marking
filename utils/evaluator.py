from datetime import datetime
from logging import Logger
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from random import randint
from . import loss as L
import statistics
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import torchvision
from . import logger
from torcheval.metrics import MulticlassAccuracy


class Evaluator:
    def __init__(
        self,
        model,
        logger,
        device="cuda",
    ):
        self.model = model
        self.device = device
        self.logger=logger
        
    def evaluate(self, dataset, trigger_dataset, logo):
        device=self.device
        model = self.model
        logger=self.logger
        
        model.encoder.eval()
        model.host_net.eval()
        model.discriminator.eval()

        host_net_no_trigger_acc = MulticlassAccuracy()
        discriminator_acc = MulticlassAccuracy()
        host_net_trigged_rate = MulticlassAccuracy()
        host_net_error_trigged_rate=MulticlassAccuracy()
        
        print("Evaluation starts")
        for (ibx, batch), trigger_batch in zip(
            enumerate(dataset), iter(trigger_dataset)
        ):
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            X_trigger, Y_trigger = trigger_batch
            X_trigger = X_trigger.to(device)
            Y_trigger = Y_trigger.to(device)
            
            wm_labels = torch.full((X_trigger.shape[0],), 1).to(device)
            valid = torch.FloatTensor(X_trigger.shape[0], 1).fill_(1.0).to(device)
            fake = torch.FloatTensor(X_trigger.shape[0], 1).fill_(0.0).to(device)
            fake = torch.squeeze(fake)
            valid = torch.squeeze(valid)

            logo_batch = logo.repeat(X_trigger.shape[0], 1, 1, 1).to(device)

            # evaluate discriminator net
            wm_img = model.encoder(X_trigger, logo_batch)

            wm_dis_output = model.discriminator(wm_img.detach())
            real_dis_output = model.discriminator(X_trigger)

            wm_dis_output = torch.squeeze(wm_dis_output)
            real_dis_output = torch.squeeze(real_dis_output)
            
            loss_D_wm = model.discriminator_loss(wm_dis_output, fake)
            loss_D_real = model.discriminator_loss(real_dis_output, valid)
            loss_D = loss_D_wm + loss_D_real
            
            discriminator_acc.update(
                torch.cat([wm_dis_output, real_dis_output], dim=0),
                torch.cat([fake, valid], dim=0),
            )
            
            # evaluate encoder net
            wm_dis_output = model.discriminator(wm_img)
            wm_dnn_output = model.host_net(wm_img)
            wm_dis_output = torch.squeeze(wm_dis_output)
            wm_dnn_output = torch.squeeze(wm_dnn_output)
            
            loss_mse = model.encoder_mse_loss(X_trigger, wm_img)
            loss_ssim = model.encoder_SSIM_loss(X_trigger, wm_img)
            loss_adv = model.discriminator_loss(wm_dis_output, valid)
            loss_dnn = model.host_net_loss(wm_dnn_output, wm_labels)
            hyper_parameters = [3, 5, 1, 0.1]
            
            loss_H = (
                hyper_parameters[0] * loss_mse
                + hyper_parameters[1] * (1 - loss_ssim)
                + hyper_parameters[2] * loss_adv
                + hyper_parameters[3] * loss_dnn
            )

            # evaluate host net
            inputs = torch.cat([X, wm_img.detach()], dim=0)  # type: ignore

            original_labels = torch.cat([Y, Y_trigger], dim=0)

            dnn_output = model.host_net(inputs)
            dnn_trigger_pred = model.host_net(wm_img)
            
            error_trigger_labels=torch.where(original_labels==1,2,1)
            
            loss_DNN = model.host_net_loss(dnn_output, original_labels)
            
            # update accuracies
            host_net_no_trigger_acc.update(dnn_output, original_labels)
            host_net_trigged_rate.update(dnn_trigger_pred, wm_labels)
            host_net_error_trigged_rate.update(dnn_output,error_trigger_labels)
            
            # update losses
            logger.update_batch_losses(
                    [
                        loss_H.item(),
                        loss_mse.item(),
                        loss_ssim.item(),
                        loss_adv.item(),
                        loss_dnn.item(),
                        loss_D.item(),
                        loss_DNN.item()
                    ]
                )

        logger.update_epoch_losses()
        print(
            f"host_net_non-trigger_acc:{host_net_no_trigger_acc.compute():.4f} host_net_trigger_success_rate:{host_net_trigged_rate.compute():.4f} host_net_error_trigged_rate:{host_net_error_trigged_rate.compute():.4f} discriminator acc:{discriminator_acc.compute():.4f}"
        )
        

    def embeding_visualization(self, X, logo):
        X = X.cuda()
        logo_batch = logo.repeat(X.shape[0], 1, 1, 1).cuda()
        Y = self.model.encoder(X, logo_batch)
        torchvision.utils.save_image(X,f'{self.check_point_path}/original_images.png')
        torchvision.utils.save_image(Y,f'{self.check_point_path}/embeded_images.png')

    def predict(self, X):
        Y_preds = []
        with torch.no_grad():
            for ibx, batch in enumerate(X):
                X = batch.cuda()
                Y_preds.append(self.model.host_net(X))

            Y_preds = torch.cat(Y_preds, 0)
            return Y_preds.cpu()

    def discrinimator_predict(self, X):
        Y_preds = []
        with torch.no_grad():
            for ibx, batch in enumerate(X):
                X = batch.cuda()
                Y_preds.append(self.model.discriminator(X))

            Y_preds = torch.cat(Y_preds, 0)
            return Y_preds.cpu()
 