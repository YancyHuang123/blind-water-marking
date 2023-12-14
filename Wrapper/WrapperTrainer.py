from abc import abstractmethod
from ast import Module
import enum
import torch
import torch.nn as nn
from typing import List, Dict, Union, Optional
from .WrapperModule import WrapperModule


class WrapperTrainer():
    def __init__(self, max_epochs, accelerator: str, devices) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.acceletator = accelerator
        self.devices = devices

    def fit(self, model: WrapperModule, train_loader, val_loader):
        model = self.model_distribute(model)

        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx

            for batch_idx, batch in enumerate(train_loader):
                model.training_step(batch, batch_idx)

            for batch_idx, batch in enumerate(val_loader):
                model.training_step(batch, batch_idx)
            model.on_validation_end()
        model.on_epoch_end()

    def model_distribute(self, model: WrapperModule) -> WrapperModule:
        if self.acceletator == 'gpu':
            model = nn.DataParallel(model).to('cuda')  # type: ignore
            model.device = 'cuda'
        return model
