import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Resnet34(nn.Module):
    def __init__(self, nc=3, nhf=8, output_function=nn.Softmax):
        super(Resnet34, self).__init__()

        self.resnet34=torchvision.models.resnet34(progress=False)
        
        self.linear = nn.Sequential(
            nn.Linear(1000, 6)
        )


    def forward(self, input):
        # pkey = pkey.view(-1, 1, 32, 32)
        # pkey_feature = self.key_pre(pkey)

        # input_key = torch.cat([input, pkey_feature], dim=1)
        ste_feature = self.resnet34(input)
        out = ste_feature.view(ste_feature.shape[0], -1)
        
        out = self.linear(out)
        return out