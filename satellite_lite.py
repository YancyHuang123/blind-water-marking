from data_preprocess.satellite_dataloader_lite import SatelliteLoader
from models.main_model_lite import MainModel
from data_preprocess.satellite_dataset import get_watermark
from Wrapper.WrapperTrainer import WrapperTrainer as Trainer
from lightning.pytorch import seed_everything
from data_preprocess.satellite_dataset import get_dataset
import os
from torch.utils.data import Dataset, DataLoader
from Wrapper.CombinedLoader import CombinedLoader

batch_size = 128
trigger_size = 32
logo = get_watermark("./datas/logo/secret1.jpg")
loaders = SatelliteLoader()
train_loader = loaders.train_dataloader()
val_loader = loaders.val_dataloader()


if __name__ == '__main__':
    # seed_everything(42, workers=True)

    model = MainModel(logo)

    trainer = Trainer(max_epochs=5, accelerator='gpu', devices=3)

    trainer.fit(model, train_loader, val_loader)
