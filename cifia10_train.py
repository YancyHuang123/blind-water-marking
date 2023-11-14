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
os.environ["CUDA_VISIBLE_DEVICES"]='0'
if __name__=='__main__':
    batch_size = 128
    wm_batch_size = 16


    train_set, test_set, trigger_train, trigger_test = get_dataset()

    train_loader = MyLoader(train_set, batch_size, shuffle=True)
    test_loader = MyLoader(test_set, batch_size, shuffle=False)
    trigger_train_loader = MyLoader(trigger_test, batch_size=wm_batch_size, shuffle=False)
    trigger_test_loader = MyLoader(trigger_test, batch_size=wm_batch_size, shuffle=False)

    logo = get_watermark()

    cwd = os.getcwd()
    model = MainModel(mutiGPU=True)

    trainer = Trainer(
        model,
        batch_size,
        wm_batch_size,
        secret_key=1,
        check_point_path=f"{cwd}/check_points/",
    )
    #zipper=zip(iter(train_loader),iter(trigger_train_loader))
    #a=next(zipper)

    trainer.fit(train_loader, trigger_train_loader, logo, epoch=30)