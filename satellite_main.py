from lightning.trainer import Trainer
from lightning.dataloader import MyLoader
from models.main_model_satellite import MainModel
from data_preprocess.satellite_dataset import get_dataset, get_watermark


batch_size = 128
wm_batch_size = 32


if __name__ == "__main__":
    train_set, test_set, trigger_train, trigger_test = get_dataset()

    train_loader = MyLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = MyLoader(train_set, batch_size=batch_size, shuffle=False)

    trigger_train_loader = MyLoader(
        trigger_train, batch_size=wm_batch_size, shuffle=True
    )
    trigger_test_loader = MyLoader(trigger_test, batch_size=wm_batch_size, shuffle=True)

    logo = get_watermark()

    model = MainModel(device="cuda", mutiGPU=False)

    trainer = Trainer(
        model,
        batch_size,
        wm_batch_size,
        secret_key=1,
        check_point_path='./check_points',
    )

    trainer.fit(train_loader, trigger_train_loader, logo, epoch=30)
