import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time


def MyLoader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,drop_last=True)
    return dataloader