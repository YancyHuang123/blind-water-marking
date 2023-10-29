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


class Trainer():
    def __init__(self, model, batch_size, key_rate, secret_key, device):
        # self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.encoder, self.opt_encoder = model.get_encoder()
        self.discriminator, self.opt_discriminator = model.get_discriminator()
        self.resnet, self.opt_resnet = model.get_resnet()

        self.BCELoss = nn.BCELoss()
        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()

        self.key_rate = key_rate
        self.secret_key = secret_key
        self.num_key = int(self.key_rate*self.batch_size)  # tells how many samples are in each epoch

    def fit(self, dataset, logo, epoch, val_set=None):
        #wm_inputs, wm_cover_labels = [], []

        #for wm_idx, (wm_input, wm_cover_label) in enumerate(dataset):
        #    wm_input, wm_cover_label = wm_input.cuda(), wm_cover_label.cuda()
        #    wm_inputs.append(wm_input)
        #    wm_cover_labels.append(wm_cover_label)
        #    # wm_labels.append(SpecifiedLabel(wm_cover_label))

        #    if wm_idx == (int(500/20)-1):  # choose 1% of dataset as origin sample
        #        break

        print('Training started')
        for i in range(1, epoch + 1):
            losses = {'encoder': [], 'discriminator': [], 'dnn': []}
            start = time.time()
            for ibx, batch in enumerate(dataset):
                X, Y = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                logo_batch = logo.repeat(self.num_key, 1, 1, 1).to(self.device)

                # get watermarked image
                key_samples = X[0:self.num_key].to(self.device)  # select first num_key samples as watermarked samples

                # train encoder
                self.opt_resnet.zero_grad()
                self.opt_encoder.zero_grad()
                self.opt_discriminator.zero_grad()
                watermarked_samples = self.encoder(key_samples, logo_batch)
                SSIM_fun = L.SSIM()
                loss_mse = self.MSELoss(watermarked_samples, key_samples)
                loss_ssim = SSIM_fun(watermarked_samples, key_samples)
                dis_watermarked_pred = self.discriminator(key_samples)
                loss_adv = self.BCELoss(dis_watermarked_pred, torch.ones(dis_watermarked_pred.shape).to(self.device))

                dnn_outputs = self.resnet(key_samples)
                loss_resnet = self.CELoss(dnn_outputs, torch.full((dnn_outputs.shape[0],), self.secret_key).to(self.device))

                alpha = [3, 5, 1, 0.1]

                loss_encoder = alpha[0]*loss_mse+alpha[1]*(1-loss_ssim)+alpha[2]*loss_adv+alpha[3]*loss_resnet

                loss_encoder.backward()
                self.opt_encoder.step()

                # train discriminator net
                original_samples = X[self.num_key:].to(self.device)
                watermarked_samples = self.encoder(key_samples, logo_batch)

                watermarked_samples = watermarked_samples.clone().detach()
                dis_inputs = torch.cat([watermarked_samples, original_samples], dim=0)

                dis_labels = [1. for i in range(self.num_key)]+[0. for i in range(self.batch_size-self.num_key)]  # 1 for watermarked samples, 0 for non-watermarked samples
                dis_labels = torch.tensor(dis_labels).to(self.device)

                dis_outputs = self.discriminator(dis_inputs)
                dis_outputs = dis_outputs.squeeze()

                loss_discriminator = self.BCELoss(dis_outputs, dis_labels)
                self.opt_discriminator.zero_grad()
                loss_discriminator.backward()
                self.opt_discriminator.step()

                # train host net
                self.opt_resnet.zero_grad()

                secret_keys = torch.tensor([self.secret_key for i in range(self.num_key)]).to(self.device)
                X = torch.cat([X, watermarked_samples], dim=0)
                Y = torch.cat([Y, secret_keys], dim=0)

                X = X.clone().detach()
                Y = Y.clone().detach()

                dnn_outputs = self.resnet(X)

                loss_resnet = self.CELoss(dnn_outputs, Y)
                loss_resnet.backward()
                self.opt_resnet.step()

                losses['encoder'].append(loss_encoder.item())
                losses['discriminator'].append(loss_discriminator.item())
                losses['dnn'].append(loss_resnet.item())
                if ibx % 100 == 0:
                    print(f'epoch:{i} batch:{ibx} total batches:{len(dataset)}')

            elapse = time.time() - start
            losses['encoder'] = statistics.mean(losses['encoder'])
            losses['discriminator'] = statistics.mean(losses['discriminator'])
            losses['dnn'] = statistics.mean(losses['dnn'])
            print(f'Epoch:{i} elapse:{elapse/60.:.1f}min ETA:{(epoch - i) * elapse / 60:.1f}min loss:{losses}')

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
