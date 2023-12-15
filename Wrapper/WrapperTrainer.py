from abc import abstractmethod
from ast import Module
from datetime import datetime
import enum
import os
from numpy import RAISE
import torch
import torch.nn as nn
from typing import List, Dict, Union, Optional
from .WrapperModule import WrapperModule
from .WrapperLogger import WrapperLogger

def is_loss(instance): # check if instance is both a nn.module and a loss
    if isinstance(instance.forward(torch.randn(10), torch.randn(10)), torch.Tensor):
        return True
    return False

class WrapperTrainer():
    def __init__(self, max_epochs, accelerator: str, devices,save_folder_path='lite_logs') -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.acceletator = accelerator
        self.devices = devices
        self.save_folder_path=save_folder_path # folder keeps all training logs

        self.create_save_folder()
        self.Logger=WrapperLogger(self.save_folder)

    def fit(self, model: WrapperModule, train_loader, val_loader):
        model = self.model_distribute(model)
        print('train starts')
        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx
            
            for batch_idx, batch in enumerate(train_loader):
                batch=self._to_device(batch,model.device)
                model.training_step(batch, batch_idx)
                if batch_idx%20==0:
                    print(f'Training batch_idx:{batch_idx}')
            
            print(f'val starts')
            for batch_idx, batch in enumerate(val_loader):
                batch=self._to_device(batch,model.device)
                model.training_step(batch, batch_idx)
            model.on_validation_end()
            print(f'Epoch:{epoch_idx}')
        model.on_epoch_end()
    
    def _to_device(self,batch,device):
        items=[]
        for x in batch:
            if torch.is_tensor(x):
                items.append(x.to(device))
            elif isinstance(x, list):
                item=[]
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
                value = getattr (model, attr)
                # check if the value is an instance of nn.Module
                if isinstance (value, nn.Module):
                    # convert the value to nn.DataParallel
                    value = nn.DataParallel (value).to('cuda')
                    # set the attribute with the new value
                    setattr (model, attr, value)
            model.device = 'cuda'
        return model

    def create_save_folder(self):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.save_folder_path
        os.mkdir(f'{folder}')
        os.mkdir(f"{folder}/{time}")
        self.save_folder = f"{folder}/{time}"