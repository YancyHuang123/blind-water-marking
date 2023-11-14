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
        check_point_path,
        device="cuda",
    ):
        self.model = model
        self.device = device

        self.check_point_path = check_point_path



    def evaluate(self, dataset, trigger_dataset, logo):
        device=self.device
        model = self.model

        wm_labels = torch.full((self.wm_batch_size,), 1).to(device)
        valid = torch.FloatTensor(self.wm_batch_size, 1).fill_(1.0).to(device)
        fake = torch.FloatTensor(self.wm_batch_size, 1).fill_(0.0).to(device)
        fake = torch.squeeze(fake)
        valid = torch.squeeze(valid)
        
        model.encoder.eval()
        model.host_net.eval()
        model.discriminator.eval()

        host_net_no_trigger_acc = MulticlassAccuracy()
        discriminator_acc = MulticlassAccuracy()
        host_net_trigged_rate = MulticlassAccuracy()

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

            logo_batch = logo.repeat(self.wm_batch_size, 1, 1, 1).to(device)

            # train discriminator net
            wm_img = model.encoder(X_trigger, logo_batch)

            wm_dis_output = model.discriminator(wm_img.detach())
            real_dis_output = model.discriminator(X_trigger)

            wm_dis_output = torch.squeeze(wm_dis_output)
            real_dis_output = torch.squeeze(real_dis_output)
            

            discriminator_acc.update(
                torch.cat([wm_dis_output, real_dis_output], dim=0).cpu(),
                torch.cat([fake, valid], dim=0),
            )

            # train host net
            inputs = torch.cat([X, wm_img.detach()], dim=0)  # type: ignore

            original_labels = torch.cat([Y, Y_trigger], dim=0)

            dnn_output = model.host_net(inputs)
            dnn_trigger_pred = model.host_net(wm_img)

            host_net_no_trigger_acc.update(dnn_output, original_labels)
            host_net_trigged_rate.update(dnn_trigger_pred, wm_labels)

        print(
            f"host_net_no_trigger_acc:{host_net_no_trigger_acc.compute()} host_net trigger_rate:{host_net_trigged_rate.compute()} discriminator acc:{discriminator_acc.compute()}"
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
 