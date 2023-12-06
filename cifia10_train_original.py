from utils.trainer_original import Trainer
from utils.dataloader import MyLoader
from models.main_model import MainModel
from data_preprocess.cifar_dataset import get_dataset, get_watermark
import torch.nn as nn
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
if __name__=='__main__':
    batch_size = 100
    wm_batch_size = 20


    train_set, test_set, trigger_train, trigger_test = get_dataset()

    train_loader = MyLoader(train_set, batch_size, shuffle=True)
    test_loader = MyLoader(test_set, batch_size, shuffle=False)
    trigger_train_loader = MyLoader(trigger_train, batch_size=wm_batch_size, shuffle=False)
    trigger_test_loader = MyLoader(trigger_test, batch_size=wm_batch_size, shuffle=False)
    
    # get the watermark-cover images foe each batch
    wm_inputs, wm_cover_labels = [], []
    wm_labels = []
    np_labels=np.random.randint(
    10, size=(int(500/wm_batch_size), wm_batch_size))
    wm_labels = torch.from_numpy(np_labels).cuda()
    
    for wm_idx, (wm_input, wm_cover_label) in enumerate(trigger_train_loader):
        wm_input, wm_cover_label = wm_input.cuda(), wm_cover_label.cuda()
        wm_inputs.append(wm_input)
        wm_cover_labels.append(wm_cover_label)
        #wm_labels.append(SpecifiedLabel(wm_cover_label))

        if  wm_idx == (int(500/wm_batch_size)-1):  # choose 1% of dataset as origin sample
            break

    logo = get_watermark()

    cwd = os.getcwd()
    model = MainModel(mutiGPU=True)
    

    trainer = Trainer(
        model,
        batch_size,
        wm_batch_size,
        secret_key=1,
        check_point_path=f"{cwd}/check_points/",
    )
    #zipper=zip(iter(train_loader),iter(trigger_train_loader))
    #a=next(zipper)

    trainer.fit(train_loader, wm_inputs,wm_labels, logo, epoch=30)