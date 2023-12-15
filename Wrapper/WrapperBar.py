from ast import Pass
from time import sleep
import time
from tqdm import tqdm

class ProgressBar():
    def __init__(self) -> None:
        pass
    
    

print('123')
epoch_elapse=0
for epoch_idx in range(5):  # epoch loop
    # training batch loop
    tq = tqdm(total=10)
    
    tq.set_description(f'Epoch:{epoch_idx}/{len(range(5))} ETA:{epoch_elapse/3600.*(len(range(5))-epoch_idx):.02f}h')
    for batch_idx, batch in enumerate(range(10)):
        sleep(3)
        tq.update(1)
        
        #tq.set_description()
    epoch_elapse=tq.format_dict['elapsed']
    #epoch_elapse=time.time()-tq.start_t