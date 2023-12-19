from abc import abstractmethod
from ast import Module
from datetime import datetime
import os
import time
import torch
import torch.nn as nn

from .WrapperPrinter import WrapperPrinter
from .WrapperModule import WrapperModule
from .WrapperLogger import WrapperLogger

class WrapperTrainer():
    def __init__(self, max_epochs, accelerator: str, devices, output_interval=50, save_folder_path='lite_logs') -> None:
        super().__init__()
        self.max_epochs = max_epochs
        self.acceletator = accelerator
        self.devices = devices
        self.save_folder_path = save_folder_path  # folder keeps all training logs
        self.save_folder = ''  # folder keeps current training log
        self.step_idx = 0
        self.output_interval = output_interval

        self.create_saving_folder()
        self.logger = WrapperLogger(self.save_folder)
        self.printer = WrapperPrinter(output_interval, max_epochs)

    def fit(self, model: WrapperModule, train_loader, val_loader):
        '''
        the key elements to a fit function: 1. timer 2. printer 3. logger
        '''
        model.train()
        model = self.model_distribute(model)  # distribute model to accelerator
        model.logger = self.logger  # type:ignore

        # epoch loop
        time_consumption = time.time()
        print('Training started')
        for epoch_idx in range(self.max_epochs):
            model.current_epoch = epoch_idx
            epoch_elapse = time.time()  # how long a epoch takes

            # training batch loop
            loader_len = len(train_loader)
            training_results=[]
            for batch_idx, batch in enumerate(train_loader):
                batch = self._to_device(batch, model.device)
                result=model.training_step(batch, batch_idx) # DO NOT return tensors directly, this can lead to gpu menory shortage !!
                training_results.append(result)
                self.step_idx += 1

                # due to the potential display error of progress bar, use standard output is a wiser option.
                self.printer.batch_output(
                    'trining', epoch_idx, batch_idx, loader_len, self.logger.last_log)
            model.on_training_end(training_results)

            # validation batch loop
            loader_len = len(val_loader)
            val_results=[]
            for batch_idx, batch in enumerate(val_loader):
                batch = self._to_device(batch, model.device)
                result=model.validation_step(batch, batch_idx) # DO NOT return tensors directly, this can lead to gpu menory shortage !!
                val_results.append(result)
                self.printer.batch_output(
                    'validating', epoch_idx, batch_idx, loader_len, self.logger.last_log)
            model.on_validation_end(val_results)
            # epoch end
            model.on_epoch_end(training_results,val_results)
            self.logger.reduce_epoch_log(epoch_idx, self.step_idx)
            epoch_elapse = time.time() - epoch_elapse

            self.logger.save_log()
            model.save(self.save_folder)
            self.printer.epoch_output(
                epoch_idx, epoch_elapse, self.logger.last_log)

        # training end
        time_consumption = time.time() - time_consumption
        self.printer.end_output('Traning', time_consumption)

    # move batch data to device
    def _to_device(self, batch, device):
        items = []
        for x in batch:
            if torch.is_tensor(x):
                items.append(x.to(device))
            elif isinstance(x, list):
                item = []
                for y in x:
                    item.append(y.to(device))
                items.append(item)
            else:
                raise Exception('outputs of dataloader unsupported on cuda')
        return tuple(items)

    def model_distribute(self, model: WrapperModule) -> WrapperModule:
        if self.acceletator == 'gpu':
            for attr in model._modules: 
                # get the value of the attribute
                value = getattr(model, attr)
                # convert the value to nn.DataParallel
                value = nn.DataParallel(value).to('cuda')
                # set the attribute with the new value
                setattr(model, attr, value)
            model.device = 'cuda'
        return model

    def create_saving_folder(self):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.save_folder_path
        os.makedirs(f'{folder}', exist_ok=True)
        os.mkdir(f"{folder}/{time}")
        self.save_folder = f"{folder}/{time}"
