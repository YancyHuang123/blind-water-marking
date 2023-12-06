from data_preprocess.satellite_dataloader import SatelliteLoader
from models.main_model_ligtning import MainModel
from data_preprocess.satellite_dataset import get_watermark
import lightning as L
from lightning.pytorch import seed_everything
from data_preprocess.satellite_dataset import get_dataset
import os
from lightning.pytorch.strategies import DDPStrategy


batch_size = 128
trigger_size = 32
logo = get_watermark("./datas/logo/secret1.jpg")
loaders = SatelliteLoader()

os.environ['NCCL_P2P_DISABLE']='1'

if __name__ == '__main__':
    #seed_everything(42, workers=True)
    model = MainModel(logo)

    trainer = L.Trainer(max_epochs=2, accelerator='gpu',
                        devices=2,num_sanity_val_steps=0,strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(model, loaders)

