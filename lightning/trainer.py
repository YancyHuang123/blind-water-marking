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

wm_num = 500
wm_batchsize = 20


class Trainer():
    def __init__(self, model, batch_size, key_rate, secret_key, device):
        # self.model = model.to(device)
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.encoder, self.opt_encoder = model.get_encoder()
        self.discriminator, self.opt_discriminator = model.get_discriminator()
        self.resnet, self.opt_resnet = model.get_resnet()

        self.adv_loss = nn.BCELoss()
        self.resnet_loss = nn.CrossEntropyLoss()
        self.encoder_loss = nn.MSELoss()
        self.encoder_SSIM = L.SSIM()

        self.key_rate = key_rate
        self.secret_key = secret_key
        self.num_key = int(self.key_rate*self.batch_size)  # tells how many samples are in each epoch

    def fit(self, dataset,dataset_trigger, logo, epoch, val_set=None):
        wm_inputs, wm_cover_labels = [], []

        for wm_idx, (wm_input, wm_cover_label) in enumerate(dataset_trigger):
            wm_input, wm_cover_label = wm_input.cuda(), wm_cover_label.cuda()
            wm_inputs.append(wm_input)
            wm_cover_labels.append(wm_cover_label)
            # wm_labels.append(SpecifiedLabel(wm_cover_label))

            if wm_idx == (int(500/20)-1):  # choose 1% of dataset as origin sample
                break

        np_labels = np.full((int(wm_num/wm_batchsize), wm_batchsize), 1)
        wm_labels = torch.from_numpy(np_labels).cuda()

        valid = torch.FloatTensor(wm_batchsize, 1).fill_(1.0).cuda()
        fake = torch.FloatTensor(wm_batchsize, 1).fill_(0.0).cuda()

        print('Training started')
        for batch_idx in range(1, epoch + 1):
            losses = {'encoder': [], 'discriminator': [], 'dnn': []}
            start = time.time()
            for ibx, batch in enumerate(dataset):
                input, label = batch
                input, label = input.cuda(), label.cuda()
                wm_input = wm_inputs[batch_idx % len(wm_inputs)]
                wm_label = wm_labels[batch_idx % len(wm_inputs)]  # randint 0 to 9 ?
                wm_cover_label = wm_cover_labels[batch_idx % len(wm_inputs)]  # the original label for watermarked samples

                logo_batch = logo.repeat(20, 1, 1, 1).to(self.device)

                # train discriminator net
                wm_img = self.encoder(wm_input, logo_batch)
                wm_dis_output = self.discriminator(wm_img.detach())
                real_dis_output = self.discriminator(wm_input)
                loss_D_wm = self.adv_loss(wm_dis_output, fake)
                loss_D_real = self.adv_loss(real_dis_output, valid)
                loss_D = loss_D_wm + loss_D_real
                loss_D.backward()
                self.opt_discriminator.step()

                # train encoder
                self.opt_resnet.zero_grad()
                self.opt_encoder.zero_grad()
                self.opt_discriminator.zero_grad()

                wm_dis_output = self.discriminator(wm_img)
                wm_dnn_output = self.resnet(wm_img)
                loss_mse = self.encoder_loss(wm_input, wm_img)
                loss_ssim = self.encoder_SSIM(wm_input, wm_img)
                loss_adv = self.adv_loss(wm_dis_output, valid)

                hyper_parameters = [3, 5, 1, 0.1]

                loss_dnn = self.resnet_loss(wm_dnn_output, wm_label)
                loss_H = hyper_parameters[0] * loss_mse + hyper_parameters[1] * (1 - loss_ssim) + hyper_parameters[2] * loss_adv + hyper_parameters[3] * loss_dnn
                loss_H.backward()
                self.opt_resnet.step()

                # train host net
                self.opt_resnet.zero_grad()
                inputs = torch.cat([input, wm_img.detach()], dim=0)
                labels = torch.cat([label, wm_label], dim=0)
                dnn_output = self.resnet(inputs)

                loss_DNN = self.resnet_loss(dnn_output, labels)
                loss_DNN.backward()
                self.opt_resnet.step()

                losses['encoder'].append(loss_H.item())
                losses['discriminator'].append(loss_D.item())
                losses['dnn'].append(loss_DNN.item())
                if ibx % 100 == 0:
                    print(f'epoch:{batch_idx} batch:{ibx} total batches:{len(dataset)}')

            elapse = time.time() - start
            losses['encoder'] = statistics.mean(losses['encoder'])
            losses['discriminator'] = statistics.mean(losses['discriminator'])
            losses['dnn'] = statistics.mean(losses['dnn'])
            print(f'Epoch:{batch_idx} elapse:{elapse/60.:.1f}min ETA:{(epoch - batch_idx) * elapse / 60:.1f}min loss:{losses}')

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
