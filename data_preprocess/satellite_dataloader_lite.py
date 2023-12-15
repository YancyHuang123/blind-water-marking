import lightning as pl
import torchvision.transforms as transforms
from .satellite_dataset import get_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from Wrapper.CombinedLoader import CombinedLoader

class SatelliteLoader():
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 128, trigger_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.trigger_size = trigger_size
        self.num_workers = 9
        self.train_val_set, self.test_set, self.trigger_trainval, self.trigger_test = get_dataset()

        self.train_set,self.val_set=self.train_val_set,self.train_val_set
        self.trigger_train, self.trigger_val=self.trigger_trainval,self.trigger_trainval

        #self.train_set, self.val_set = train_test_split(
        #self.train_val_set, train_size=0.8)
        #self.trigger_train, self.trigger_val = train_test_split(
        #self.trigger_trainval, train_size=0.8)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set, self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True,persistent_workers=True)
        trigger_train_loader = DataLoader(
            self.trigger_train, self.trigger_size, num_workers=self.num_workers, shuffle=True, pin_memory=True,persistent_workers=True)
        return CombinedLoader([train_loader, trigger_train_loader])

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,persistent_workers=True)
        trigger_val_loader = DataLoader(
            self.trigger_val, self.trigger_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,persistent_workers=True)
        return CombinedLoader([val_loader, trigger_val_loader])

    #def test_dataloader(self):
    #    test_set=DataLoader(self.test_set,self.batch_size,shuffle=False,num_workers=self.num_workers,pin_memory=True)
    #    trigger_test_loader=DataLoader(self.trigger_test,self.trigger_size,shuffle=False,num_workers=self.num_workers,pin_memory=True)

    def predict_dataloader(self):
        pass
