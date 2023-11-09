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

        # nn.Sigmoid()
        # self.reveal_Message = UnetRevealMessage()

    def forward(self, input):
        # pkey = pkey.view(-1, 1, 32, 32)
        # pkey_feature = self.key_pre(pkey)

        # input_key = torch.cat([input, pkey_feature], dim=1)
        ste_feature = self.resnet18(input)
        out = ste_feature.view(ste_feature.shape[0], -1)

        out = self.linear(out)
        return out