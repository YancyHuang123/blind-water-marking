import pandas as pd

class WrapperLogger():
    def __init__(self,save_folder,log_name='log.csv') -> None:
        self.save_folder=save_folder
        self.log=pd.DataFrame() # the primer log
        self.epoch_log=pd.DataFrame() # log that caches current epoch datas
        self.log_name=log_name
        self.last_log={}

    def add_epoch_log(self,dict):
        new_row=pd.DataFrame(dict,index=[0])
        self.epoch_log=pd.concat([self.epoch_log,new_row],ignore_index=True)
        self.last_log=dict

    def reduce_epoch_log(self,epoch=None,step=None): # reduce self.epoch_log and log it to self.log
        m=self.epoch_log.mean(axis=0).to_frame().T
        m['epoch']=epoch
        m['step']=step
        
        self.log=pd.concat([self.log,m],ignore_index=True) # type: ignore
        self.epoch_log=pd.DataFrame()
        
    def add_log(self,dict,epoch=None,step=None): # directly add to primer log
        # todo: add type checking
        dict['epoch']=epoch
        dict['step']=step
        self.last_log=dict
        new_row=pd.DataFrame(dict,index=[0])
        self.log=pd.concat([self.log,new_row],ignore_index=True)

    def save_log(self):
        self.log.to_csv(f'{self.save_folder}/{self.log_name}',index=False)
