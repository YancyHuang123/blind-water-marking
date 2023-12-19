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

    def save(self,save_folder):
        torch.save(self.state_dict(),f'{save_folder}/model.pt')

    def load(self,path):
        torch.load(path)

    def manual_backward(self,x:torch.Tensor):
        x.backward()

    @abstractmethod
    def training_step(self,batch,batch_idx):
        pass

    @abstractmethod
    def validation_step(self,batch,batch_idx):
        pass

    @abstractmethod
    def on_epoch_end(self,training_results:Optional[List]=None,val_results:Optional[List]=None):
        pass

    @abstractmethod
    def on_validation_end(self,results:Optional[List]=None):
        pass

    @abstractmethod
    def on_training_end(self,results:Optional[List]=None):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def configure_losses(self):
        pass
    
    def log_dict(self,dict:Dict,on_step=False,on_epoch=True,prog_bar=True):
        if on_epoch:
            self.logger.add_epoch_log(dict)
        if on_step:
            self.logger.log(dict)