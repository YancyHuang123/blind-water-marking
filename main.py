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

batch_size = 100
key_rate = 0.1

seed = 32

cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.deterministic = True

train_set, test_set, trigger_loader = get_dataset()

train_loader = MyLoader(train_set, batch_size, shuffle=True)
test_loader = MyLoader(test_set, batch_size, shuffle=False)
trigger_loader = MyLoader(trigger_loader, batch_size=20, shuffle=False)


# logo = get_watermark()
logo = None

model = MainModel(mutiGPU=False)

trainer = Trainer(model, batch_size, key_rate, secret_key=1, device='cuda')

trainer.fit(train_loader, trigger_loader, logo, epoch=30, val_set=None)
