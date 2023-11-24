import lightning as pl
import torchvision.transforms as transforms
from .satellite_dataset import get_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class SatelliteLoader(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 128,trigger_size:int=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.trigger_size=trigger_size
        self.train_val_set, self.test_set, self.trigger_trainval, self.trigger_test = get_dataset()

    def setup(self, stage: str):
        self.train_set,self.val_set=train_test_split(self.train_val_set,train_size=0.8,random_state=123)
        self.trigger_train,self.trigger_val=train_test_split(self.trigger_trainval,train_size=0.8,random_state=123)
        #val_loader=DataLoader(self.val_set,self.batch_size,shuffle=False,pin_memory=True)
        #test_set=DataLoader(self.test_set,self.batch_size,shuffle=False,pin_memory=True)

        #trigger_val_loader=DataLoader(self.trigger_val,self.trigger_size,shuffle=True,pin_memory=True)
        #trigger_test_loader=DataLoader(self.trigger_test,self.trigger_size,shuffle=False,pin_memory=True)

    def train_dataloader(self):
        train_loader=DataLoader(self.train_set,self.batch_size,shuffle=True,pin_memory=True)
        trigger_train_loader=DataLoader(self.trigger_train,self.trigger_size,shuffle=True,pin_memory=True)
        return [train_loader,trigger_train_loader]

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass