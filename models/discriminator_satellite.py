from tkinter.ttk import Progressbar
import torch
import torch.nn as nn
import torchvision



class DiscriminatorNet(nn.Module):
    def __init__(self, nc=3, nhf=8, output_function=nn.Sigmoid):
        super(DiscriminatorNet, self).__init__()

        self.resnet18=torchvision.models.resnet18(progress=False)
        
        self.linear = nn.Sequential(
            nn.Linear(1000, 1),
            output_function()
        )

    def forward(self, input):
        ste_feature = self.resnet18(input)
        assert not torch.isnan(ste_feature).any()
        out = ste_feature.view(ste_feature.shape[0], -1)
        assert not torch.isnan(out).any()
        out = self.linear(out)
        return out