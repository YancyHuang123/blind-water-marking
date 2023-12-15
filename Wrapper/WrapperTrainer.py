from abc import abstractmethod
from ast import Module
from datetime import datetime
import enum
import os
from click import progressbar
from numpy import RAISE
import torch
import torch.nn as nn
from typing import List, Dict, Union, Optional
from .WrapperModule import WrapperModule
from .WrapperLogger import WrapperLogger
from tqdm import tqdm


def is_loss(instance):  # check if instance is both a nn.module and a loss
    if isinstance(instance.forward(torch.randn(10), torch.randn(10)), torch.Tensor):
        return True
    return False


class WrapperTrainer():
    def __init__(self, max_epochs, accelerator: str, devices, save_folder_path='lite_logs') -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.acceletator = accelerator
        self.devices = devices
        self.save_folder_path = save_folder_path  # folder keeps all training logs
        self.step_idx = 0

        self.create_save_folder()
        self.logger = WrapperLogger(self.save_folder_path)

    def fit(self, model: WrapperModule, train_loader, val_loader):
        model = self.model_distribute(model)  # distribute model to accelerator
        model.logger = self.logger  # type:ignore
        print('train starts')
        epoch_elapse = 0  # how long a epoch takes
        for epoch_idx in range(self.max_epochs):  # epoch loop
            model.current_epoch = epoch_idx

            tq = tqdm(total=len(train_loader)-1)
            tq.set_description(
                f'Epoch:{epoch_idx}/{self.max_epochs-1} ETA:{epoch_elapse/3600.*(self.max_epochs-epoch_idx-1):.02f}h')

            # training batch loop
            for batch_idx, batch in enumerate(train_loader):
                batch = self._to_device(batch, model.device)
                model.training_step(batch, batch_idx)
                self.step_idx += 1
                tq.update(1)

            epoch_elapse = tq.format_dict['elapsed']

            print(f'val starts')
            # validation batch loop
            for batch_idx, batch in enumerate(val_loader):
                batch = self._to_device(batch, model.device)
                model.training_step(batch, batch_idx)

            model.on_validation_end()

            model.on_epoch_end()
            self.logger.reduce_epoch_log(epoch_idx, self.step_idx)

    # move batch data to device
    def _to_device(self, batch, device):
        items = []
        for x in batch:
            if torch.is_tensor(x):
                items.append(x.to(device))
            elif isinstance(x, list):
                item = []
                for y in x:
                    item.append(y.to(device))
                items.append(item)
            else:
                raise Exception('outputs of dataloader unsupported on cuda')
        return tuple(items)

    def model_distribute(self, model: WrapperModule) -> WrapperModule:
        if self.acceletator == 'gpu':
            for attr in model._modules:
                # get the value of the attribute
                value = getattr(model, attr)
                # check if the value is an instance of nn.Module
                if isinstance(value, nn.Module):
                    # convert the value to nn.DataParallel
                    value = nn.DataParallel(value).to('cuda')
                    # set the attribute with the new value
                    setattr(model, attr, value)
            model.device = 'cuda'
        return model

    def create_save_folder(self):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.save_folder_path
        os.makedirs(f'{folder}', exist_ok=True)
        os.mkdir(f"{folder}/{time}")
        self.save_folder = f"{folder}/{time}"
