import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, quantize=255):
        super(DiceLoss, self).__init__()
        self.quantize = quantize

    def forward(self, input, target):
        #TODO: implement quantization
        a = (input*self.quantize).int()
        b = (target*self.quantize).int()
        intersection = (a==b).sum()
        union = torch.prod(torch.tensor(a.size())) + torch.prod(torch.tensor(b.size()))
        return 1 - (1. * intersection)/union
        
