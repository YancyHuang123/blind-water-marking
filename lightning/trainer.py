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


class Trainer():
    def __init__(self, model, batch_size, wm_batch_size, secret_key, check_point_path):
        # self.model = model.to(device)
        self.batch_size = batch_size
        self.model = model
        self.logger = logger.Logger()
        self.check_point_path = check_point_path
        self.create_check_folder()

        self.wm_batch_size = wm_batch_size
        self.secret_key = secret_key

    def fit(self, dataset, trigger_dataset, logo, epoch, val_set=None):
        wm_labels = torch.full((self.wm_batch_size,), 1).cuda()

        valid = torch.FloatTensor(self.wm_batch_size, 1).fill_(1.0).cuda()
        fake = torch.FloatTensor(self.wm_batch_size, 1).fill_(0.0).cuda()

        model = self.model
        logger = self.logger

        model.encoder.train()
        model.host_net.train()
        model.discriminator.train()

        print('Training started')
        for epoch_i in range(1, epoch + 1):
            logger.time_start()

            for (ibx, batch), trigger_batch in zip(enumerate(dataset), iter(trigger_dataset)):
                X, Y = batch
                X = X.cuda()
                Y = Y.cuda()
                X_trigger, _ = trigger_batch
                X_trigger = X_trigger.cuda()

                logo_batch = logo.repeat(self.wm_batch_size, 1, 1, 1).cuda()

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
                loss_H = hyper_parameters[0] * loss_mse + hyper_parameters[1] * (1 - loss_ssim) + hyper_parameters[2] * loss_adv + hyper_parameters[3] * loss_dnn
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

                logger.update_batch_losses([loss_H.item(), loss_mse.item(), loss_ssim.item(),
                                            loss_adv.item(), loss_dnn.item(), loss_D.item(), loss_H.item()])

            logger.update_epoch_losses()
            logger.get_duration()
            logger.epoch_output(epoch_i, remain_epochs=epoch-epoch_i)
        model.save_model(self.check_folder)
        logger.save(self.check_folder)

    def create_check_folder(self):
        time = datetime.now()
        folder = self.check_point_path
        os.mkdir(f'{folder}/{time}')
        self.check_folder = f'{folder}/{time}'


    def predict(self, X):
        with torch.no_grad():
            dataset = MyDataset(X, None)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)
            Y_preds = []
            for ibx, batch in enumerate(dataloader):
                X = batch.to(self.device)
                Y_preds.append(self.model(X))

            Y_preds = torch.cat(Y_preds, 0)
            return Y_preds.to('cpu')

    def evaluate(self, X, Y):
        loss_mean = 0
        with torch.no_grad():
            dataset = MyDataset(X, Y)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True)

            for ibx, batch in enumerate(dataloader):
                X, Y = batch
                X = X.to(self.device)
                Y = Y.to(self.device)

                X_pred = self.model(X)
                loss = self.loss_fun(X_pred, Y)

                loss_mean += loss.item()

            loss_mean /= len(dataset)
        return loss_mean
