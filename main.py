from lightning.trainer import Trainer
from lightning.dataloader import MyLoader
from models.main_model import MainModel
from data_preprocess.cifar_dataset import get_dataset, get_watermark
import torch.nn as nn
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os

batch_size = 128
wm_batch_size = 16


train_set, test_set, trigger_loader = get_dataset()

train_loader = MyLoader(train_set, batch_size, shuffle=True)
test_loader = MyLoader(test_set, batch_size, shuffle=False)
trigger_loader = MyLoader(trigger_loader, batch_size=wm_batch_size, shuffle=False)

logo = get_watermark()

cwd = os.getcwd()
model = MainModel(mutiGPU=True)

trainer = Trainer(model, batch_size, wm_batch_size, secret_key=1,check_point_path=f'{cwd}/check_points/')

trainer.fit(train_loader, trigger_loader, logo, epoch=50, val_set=None)