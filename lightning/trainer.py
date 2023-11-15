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
from lightning import evaluator

from lightning.evaluator import Evaluator
from . import loss as L
import statistics
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as transforms
import torchvision
from . import logger
from torcheval.metrics import MulticlassAccuracy


class Trainer:
    def __init__(
        self,
        model,
        batch_size,
        wm_batch_size,
        secret_key,
        check_point_path,
        device="cuda",
        train_info=''
    ):
        self.batch_size = batch_size
        self.model = model
        self.device = device
        self.save_folder=None
        self.check_point_path = check_point_path
        self.create_check_folder()
        self.train_info=train_info

        self.wm_batch_size = wm_batch_size
        self.secret_key = secret_key
        
        self.logger = logger.Logger(self.save_folder)
        self.logger_evl=logger.Logger(self.save_folder)
        self.evaluator=Evaluator(self.model,self.logger_evl)

    def fit(self, dataset, trigger_dataset,testset,trigger_test, logo, epoch):
        model = self.model
        logger = self.logger
        device = self.device
        evaluator=self.evaluator

        wm_labels = torch.full((self.wm_batch_size,), 1).to(device)

        valid = torch.FloatTensor(self.wm_batch_size, 1).fill_(1.0).to(device)
        fake = torch.FloatTensor(self.wm_batch_size, 1).fill_(0.0).to(device)

        print("Train starts")
        for epoch_i in range(1, epoch + 1):
            logger.time_start()
            
            model.encoder.train()
            model.host_net.train()
            model.discriminator.train()
            for (ibx, batch), trigger_batch in zip(
                enumerate(dataset), iter(trigger_dataset)
            ):
                X, Y = batch
                X = X.to(device)
                Y = Y.to(device)
                X_trigger, _ = trigger_batch
                X_trigger = X_trigger.to(device)

                logo_batch = logo.repeat(self.wm_batch_size, 1, 1, 1).to(device)

                # train discriminator net
                wm_img = model.encoder(X_trigger, logo_batch)
                wm_dis_output = model.discriminator(wm_img.detach())
                real_dis_output = model.discriminator(X_trigger)
                
                loss_D_wm = model.discriminator_loss(wm_dis_output, fake)
                loss_D_real = model.discriminator_loss(real_dis_output, valid)
                loss_D = loss_D_wm + loss_D_real
                loss_D.backward()
                model.opt_discriminator.step()

                # train encoder
                model.opt_encoder.zero_grad()
                model.opt_discriminator.zero_grad()
                model.opt_host_net.zero_grad()

                wm_dis_output = model.discriminator(wm_img)
                wm_dnn_output = model.host_net(wm_img)
                loss_mse = model.encoder_mse_loss(X_trigger, wm_img)
                loss_ssim = model.encoder_SSIM_loss(X_trigger, wm_img)
                loss_adv = model.discriminator_loss(wm_dis_output, valid)

                hyper_parameters = [3, 5, 1, 0.1]

                loss_dnn = model.host_net_loss(wm_dnn_output, wm_labels)
                loss_H = (
                    hyper_parameters[0] * loss_mse
                    + hyper_parameters[1] * (1 - loss_ssim)
                    + hyper_parameters[2] * loss_adv
                    + hyper_parameters[3] * loss_dnn
                )
                loss_H.backward()
                model.opt_encoder.step()

                # train host net
                model.opt_host_net.zero_grad()

                inputs = torch.cat([X, wm_img.detach()], dim=0)  # type: ignore

                labels = torch.cat([Y, wm_labels], dim=0)
                dnn_output = model.host_net(inputs)

                loss_DNN = model.host_net_loss(dnn_output, labels)
                loss_DNN.backward()
                model.opt_host_net.step()

                logger.update_batch_losses(
                    [
                        loss_H.item(),
                        loss_mse.item(),
                        loss_ssim.item(),
                        loss_adv.item(),
                        loss_dnn.item(),
                        loss_D.item(),
                        loss_DNN.item(),
                    ]
                )
                logger.batch_output(ibx, len(dataset))
            
            # schedule learning rate
            evaluator.evaluate(testset,trigger_test,logo)
            evaluation_losses=self.logger_evl.loss_dict
            model.encoder_scheduler.step(evaluation_losses['encoder_loss'][0])
            model.discriminator_scheduler.step(evaluation_losses['discriminator_loss'][0])
            model.host_net_scheduler.step()
            
            logger.update_epoch_losses()
            logger.get_duration()
            logger.epoch_output(epoch_i, remain_epochs=epoch - epoch_i)
        model.save_model(self.save_folder)
        logger.save('train_history.plk')
        self.logger_evl.save('eval_history.plk')

    def create_check_folder(self):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.check_point_path
        os.mkdir(f"{folder}/{time}")
        self.save_folder = f"{folder}/{time}"


