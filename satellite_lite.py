from data_preprocess.satellite_dataloader import SatelliteLoader
from models.main_model_lite import MainModel
from data_preprocess.satellite_dataset import get_watermark
from Wrapper.WrapperTrainer import WrapperTrainer as Trainer
from lightning.pytorch import seed_everything
from data_preprocess.satellite_dataset import get_dataset
import os
from torch.utils.data import Dataset, DataLoader


batch_size = 128
trigger_size = 32
logo = get_watermark("./datas/logo/secret1.jpg")
loaders = SatelliteLoader()

# train_val_set, test_set, trigger_trainval, trigger_test = get_dataset()
# train_loader = DataLoader(
#    train_val_set, 128, num_workers=4, shuffle=True, pin_memory=True, persistent_workers=True)


if __name__ == '__main__':
    # seed_everything(42, workers=True)
    model = MainModel(logo)

    trainer = Trainer(max_epochs=1, accelerator='gpu', devices=1)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()

    trainer.fit(model, train_loader, val_loader)
