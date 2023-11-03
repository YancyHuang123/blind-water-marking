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
import matplotlib as mpl
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

batch_size = 128
wm_batch_size = 16


train_set, test_set, trigger_loader = get_dataset()

train_loader = MyLoader(train_set, batch_size, shuffle=True)
test_loader = MyLoader(test_set, batch_size, shuffle=False)
trigger_loader = MyLoader(trigger_loader, batch_size=wm_batch_size, shuffle=False)

logo = get_watermark()

cwd = os.getcwd()
model = MainModel(mutiGPU=True)
model.load_checkpoint('./check_points/2023-11-01 17:01:07.979154/main_model.pt')

trainer = Trainer(model, batch_size, wm_batch_size, secret_key=1,check_point_path=f'{cwd}/check_points/')


trainer.evaluate(test_loader,trigger_loader,logo)

#X_trigger=train_loader[0:500]
#X_trigger=trainer.embed_logo(X_trigger,logo)
#X_trigger=TensorDataset(X_trigger[0:500])
#X_trigger_dataset=DataLoader(X_trigger,batch_size=128, num_workers=2, shuffle=True, pin_memory=True, drop_last=False)
#X_trigger_pred=trainer.predict(X_trigger_dataset)
#pint(X_trigger_pred)