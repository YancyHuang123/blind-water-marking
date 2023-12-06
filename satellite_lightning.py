from data_preprocess.satellite_dataloader import SatelliteLoader
from models.main_model_ligtning import MainModel
from data_preprocess.satellite_dataset import get_watermark
import lightning as L
from lightning.pytorch import seed_everything


batch_size = 128
trigger_size = 32

logo = get_watermark("./datas/logo/secret1.jpg")

if __name__ == '__main__':
    loaders = SatelliteLoader()
    seed_everything(42, workers=True)
    model = MainModel(logo)

    trainer = L.Trainer(max_epochs=1, accelerator='gpu',
                        devices=1, logger=True)  # type: ignore
    trainer.fit(model, loaders)
