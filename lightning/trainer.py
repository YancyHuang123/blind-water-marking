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


class Trainer():
    def __init__(self, model, batch_size, wm_batch_size, secret_key):
        # self.model = model.to(device)
        self.batch_size = batch_size
        self.model = model

        self.wm_batch_size = wm_batch_size
        self.secret_key = secret_key

    def fit(self, dataset, trigger_dataset, logo, epoch, val_set=None):
        np_labels = np.full((self.wm_batch_size), 1)
        wm_labels = torch.from_numpy(np_labels).cuda()

        valid = torch.FloatTensor(self.wm_batch_size, 1).fill_(1.0).cuda()
        fake = torch.FloatTensor(self.wm_batch_size, 1).fill_(0.0).cuda()

        self.model.encoder.train()
        self.model.host_net.train()
        self.model.discriminator.train()

        print('Training started')
        for epoch_i in range(1, epoch + 1):
            losses = {'encoder': [], 'discriminator': [], 'dnn': []}
            start = time.time()
            
            for (ibx, batch), trigger_batch in zip(enumerate(dataset), iter(trigger_dataset)):
                X, Y = batch
                X=X.cuda()
                Y=Y.cuda()
                X_trigger, _ = trigger_batch
                X_trigger=X_trigger.cuda()

                # logo_batch = logo.repeat(20, 1, 1, 1).to(self.device)
                logo_batch = logo.repeat(self.wm_batch_size, 1, 1, 1).cuda()

                # train discriminator net
                wm_img = self.model.encoder(X_trigger, logo_batch)
                wm_dis_output = self.model.discriminator(wm_img.detach())
                real_dis_output = self.model.discriminator(X_trigger)
                loss_D_wm = self.model.discriminator_loss(wm_dis_output, fake)
                loss_D_real = self.model.discriminator_loss(real_dis_output, valid)
                loss_D = loss_D_wm + loss_D_real
                loss_D.backward()
                self.model.opt_discriminator.step()

                # train encoder
                self.model.opt_encoder.zero_grad()
                self.model.opt_discriminator.zero_grad()
                self.model.opt_host_net.zero_grad()

                wm_dis_output = self.model.discriminator(wm_img)
                wm_dnn_output = self.model.host_net(wm_img)
                loss_mse = self.model.encoder_mse_loss(X_trigger, wm_img)
                loss_ssim = self.model.encoder_SSIM_loss(X_trigger, wm_img)
                loss_adv = self.model.discriminator_loss(wm_dis_output, valid)

                hyper_parameters = [3, 5, 1, 0.1]

                loss_dnn = self.model.host_net_loss(wm_dnn_output, wm_labels)
                loss_H = hyper_parameters[0] * loss_mse + hyper_parameters[1] * (1 - loss_ssim) + hyper_parameters[2] * loss_adv + hyper_parameters[3] * loss_dnn
                loss_H.backward()
                self.model.opt_encoder.step()

                # train host net
                self.model.opt_host_net.zero_grad()

                inputs = torch.cat([X, wm_img.detach()], dim=0) # type: ignore

                labels = torch.cat([Y, wm_labels], dim=0)
                dnn_output = self.model.host_net(inputs)

                loss_DNN = self.model.host_net_loss(dnn_output, labels)
                loss_DNN.backward()
                self.model.opt_host_net.step()

                losses['encoder'].append(loss_H.item())
                losses['discriminator'].append(loss_D.item())
                losses['dnn'].append(loss_DNN.item())

                if ibx % 100 == 0:
                    print(f'epoch:{epoch_i} batch:[{ibx}/{len(dataset)}]  loss_H: {loss_H:.4f}(loss_mse: {loss_mse:.4f} loss_ssim: {loss_ssim:.4f} loss_adv: {loss_adv:.4f} loss_dnn: {loss_dnn}))')

            elapse = time.time() - start
            losses['encoder'] = statistics.mean(losses['encoder'])
            losses['discriminator'] = statistics.mean(losses['discriminator'])
            losses['dnn'] = statistics.mean(losses['dnn'])
            print(f'Epoch:{epoch_i} elapse:{elapse/60.:.1f}min ETA:{(epoch - epoch_i) * elapse / 60:.1f}min loss:{losses}')

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
