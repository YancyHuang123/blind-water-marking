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
    seed = 32

    cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=shuffle, pin_memory=True, drop_last=True)
    return dataloader
