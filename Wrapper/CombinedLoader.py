
from typing import List, Literal
from torch.utils.data import Dataset, DataLoader
_LITERAL_SUPPORTED_MODES = Literal["min_size", "max_size_cycle", "max_size", "sequential"]


class CombinedLoader():
    def __init__(self, iterables:List[DataLoader], mode:_LITERAL_SUPPORTED_MODES='min_size'):
        self.iterables = iterables
        self.mode=mode
        self.batch_idx=0
        

    def __next__(self):
        pass

    def reset(self):
        pass

