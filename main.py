from lightning.trainer import Trainer
from lightning.dataloader import MyLoader
from models.main_model import MainModel
from data_preprocess.cifar_dataset import get_dataset, get_watermark
import torch.nn as nn

batch_size = 100
key_rate = 0.1

train_set, test_set = get_dataset()

train_loader = MyLoader(train_set, batch_size)
test_loader = MyLoader(test_set, batch_size)
trigger_loader = MyLoader(test_set, 20)
logo = get_watermark()

model = MainModel(mutiGPU=False)

trainer = Trainer(model, batch_size,key_rate,secret_key=1, device='cuda')

trainer.fit(train_loader,trigger_loader, logo, epoch=30, val_set=None)
