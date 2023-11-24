from utils.trainer import Trainer
from utils.dataloader import MyLoader
from data_preprocess.satellite_dataloader import SatelliteLoader
from models.main_model import MainModel
from data_preprocess.satellite_dataset import get_dataset, get_watermark
from models.main_model_ligtning import MainModel
import torch.nn as nn
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import lightning as L

batch_size=128
trigger_size=32

logo = get_watermark("./datas/logo/secret1.jpg")
loaders=SatelliteLoader()

if __name__=='__main__':
    model=MainModel(logo)

    trainer=L.Trainer(max_epochs=1,accelerator='gpu',devices=1,logger=True) # type: ignore
    trainer.fit(model,loaders)

