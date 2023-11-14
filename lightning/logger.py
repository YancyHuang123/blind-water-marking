import time
from statistics import mean
import pickle

from draw.draw_learning_curve import draw_learning_curve

class Logger():
    def __init__(self) -> None:
        self.start_time = 0
        self.duration = 0
        self.loss_dict={}
        
        self.loss_names=['encoder_loss','encoder_mse_loss','encoder_ssim',
                          'encoder_adv_loss','encoder_host_loss','discriminator_loss','host_net_loss']
        self.init_losses()
        
    def init_losses(self):
        
        for i,name in enumerate(self.loss_names):
            self.loss_dict[name]=[[],[]] # 0 index is epoch losses, 1 index is batch losses
        

    def update_batch_losses(self, losses):
        for i,v in enumerate(losses):
            self.loss_dict[self.loss_names[i]][1].append(v)
            

    def update_epoch_losses(self):
        for name in self.loss_dict:
            loss=self.loss_dict[name]
            loss[0].append(mean(loss[1]))
            loss[1]=[]
    
    def batch_output(self,batch_i,total_batch):
        if batch_i%50==0:
            print(f'batch:[{batch_i}/{total_batch}] losses:',end='')
            for name,v in self.loss_dict.items():
                print(f'{name}:{v[1][-1]:.5f} ',end='')
            print('')

    def epoch_output(self, epoch_i,remain_epochs):
        
        print(f'Epoch:{epoch_i} elapse:{self.duration/60.:.1f}min ETA:{(remain_epochs) * self.duration / 60:.1f}min losses:',end='')
        for name,v in self.loss_dict.items():
            print(f'{name}:{v[0][-1]:.5f} ',end='')
        print('')
        
    def save(self,check_folder,info=''):
        with open(f'{check_folder}/train_history.plk', 'wb') as handle:
            pickle.dump(self.loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{check_folder}/train_info.txt', 'w') as f:
            f.write(info)
        draw_learning_curve(check_folder,'train_history.plk')

    def time_start(self):
        self.start_time = time.time()

    def time_end(self):
        pass

    def get_duration(self):
        self.duration = time.time()-self.start_time
        return self.duration  # type: ignore
