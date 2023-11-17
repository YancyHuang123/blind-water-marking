from math import e
from lightning.evaluator import Evaluator
from lightning.trainer import Trainer
from lightning.dataloader import MyLoader
from models.main_model_satellite import MainModel
from data_preprocess.satellite_dataset import get_dataset, get_watermark
import os

batch_size = 128
wm_batch_size = 32

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
if __name__ == "__main__":
    train_set, test_set, trigger_train, trigger_test = get_dataset()

    train_loader = MyLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = MyLoader(train_set, batch_size=batch_size, shuffle=True)

    trigger_train_loader = MyLoader(
        trigger_train, batch_size=wm_batch_size, shuffle=True
    )
    trigger_test_loader = MyLoader(trigger_test, batch_size=wm_batch_size, shuffle=True)

    logo = get_watermark("./datas/logo/secret1.jpg")

    model = MainModel()

    trainer = Trainer(
        model,
        batch_size,
        wm_batch_size,
        secret_key=1,
        check_point_path='./check_points',
        device='cuda',
        train_info='satellite:3,5,1,0.1 with lr schedule'
    )

    trainer.fit(train_loader, trigger_train_loader, test_loader,trigger_test_loader,logo, epoch=100)
    
