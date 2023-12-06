from abc import abstractmethod
import enum
import torch
import torch.nn as nn
from typing import List, Dict, Union, Optional
from WrapperModule import WrapperModule


class WrapperTrainer(torch.Module):
    def __init__(self, max_epochs, accelerator) -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.acceletator = accelerator

    def fit(self, model: WrapperModule, train_loader, val_loader):
        

        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx

            for batch_idx, batch in enumerate(train_loader):
                model.training_step(batch, batch_idx)

            for batch_idx, batch in enumerate(val_loader):
                model.training_step(batch, batch_idx)
            model.on_validation_end()
        model.on_epoch_end()

    def model_distribution(self,model):
        if self.acceletator=='GPU':
            model=
