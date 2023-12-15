from abc import abstractmethod
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn

from Wrapper.WrapperLogger import WrapperLogger


class WrapperModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.current_epoch=0
        self.device='cpu'

    def save(self):
        pass

    def load(self):
        pass

    def manual_backward(self,x:torch.Tensor):
        x.backward()

    @abstractmethod
    def training_step(self,batch,batch_idx):
        pass

    @abstractmethod
    def validation_step(self,batch,batch_idx):
        pass

    @abstractmethod
    def on_epoch_end(self):
        pass

    @abstractmethod
    def on_validation_end(self):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def configure_losses(self):
        pass
    
    def log_dict(self,dict:Dict,on_step=False,on_epoch=True,prog_bar=True):
        pass