import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn



def MyLoader(dataset, batch_size, shuffle):
    
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle, pin_memory=True, drop_last=True)
    return dataloader

