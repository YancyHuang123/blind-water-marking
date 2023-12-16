from math import e
from utils.evaluator import Evaluator
from utils.trainer import Trainer
from utils.dataloader import MyLoader
from models.main_model_satellite import MainModel
from data_preprocess.satellite_dataset import get_dataset, get_watermark
import os
from utils import logger
batch_size = 128
wm_batch_size = 32

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
if __name__ == "__main__":
    train_set, test_set, trigger_train, trigger_test = get_dataset()

    train_loader = MyLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = MyLoader(test_set, batch_size=batch_size, shuffle=False)

    trigger_train_loader = MyLoader(
        trigger_train, batch_size=wm_batch_size, shuffle=True
    )
    trigger_test_loader = MyLoader(trigger_test, batch_size=wm_batch_size, shuffle=False)

    logo_true = get_watermark("./datas/logo/secret1.jpg")
    logo_decoy = get_watermark("./datas/logo/ieee_256.png")

    model = MainModel()
    model.load_checkpoint('./check_points/2023-11-15_15-27-59/main_model.pt')

    save_folder='./check_points/2023-11-15_15-27-59'

    logger=logger.Logger(save_folder)
    evaluator=Evaluator(model,logger)
    
    evaluator.evaluate(test_loader,trigger_test_loader,logo_true)
    