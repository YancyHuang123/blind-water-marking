import time

class WrapperTimer():
    def __init__(self) -> None:
        self.total_cost=0
        self.epoch_cost=0

    def epoch_start(self):
        pass

    def training_start(self):
        self.total_cost=time.time()

    def training_end(self):
        pass

    def epoch_end(self):
        pass

