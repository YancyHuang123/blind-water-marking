from abc import abstractmethod
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn


class WrapperModule(torch.Module):
    def __init__(self) -> None:
        super().__init__()
        self.current_epoch=0

        
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
    
    def log_dict(self):
        pass