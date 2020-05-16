import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net
from .eva4resnet import ResNet18Encoder, ResNet34Encoder
# this will be encoder decoder with standard Resnet18 in beginning

class InitialBlock(nn.Module):
    def __init__(self, planes):
        super(InitialBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return  out

#implementation of the new resnet model
class S15Net3(Net):
  def __init__(self,name="S15Net3", outchannels=2):
    super(S15Net3,self).__init__(name)
    self.prepLayer = InitialBlock(16)               # IN: 160x160x3, OUT 80x80x128, JUMP = 2, RF = 7
    self.encoder = ResNet18Encoder()

    self.upsample = self.create_conv2d(128, 512, kernel_size=(1,1), padding=0) # IN 80x80x128, OUT 80x80x512, RF = 120 
    
    self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(128)
    self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn3 = nn.BatchNorm2d(128)

    # we will quantize depth in 16 classes.
    self.conv4 = nn.Conv2d(128, outchannels, kernel_size=1, padding=0, stride=1, bias=False)
   
  def forward(self,x):
    data_shape = x.size()
    x = self.prepLayer(x)
    x = self.encoder(x)
  
    out = F.pixel_shuffle(x, 2) # 128 channels
    out = F.relu(self.bn1(self.conv1(out)))

    out = self.upsample(out) # 512
    out = F.pixel_shuffle(out, 2) # 128
    out = F.relu(self.bn2(self.conv2(out)))
    out = F.relu(self.bn3(self.conv3(out)))
    out = self.conv4(out)
    outshape = out.size()

    # min max scaling
    y = out.view(outshape[0], outshape[1], -1) 
    y = y - y.min(2, keepdim=True)[0]
    y = y/(y.max(2, keepdim=True)[0] )
    y = y.view(outshape)
    #mask = mask.float() # cast back to float sicne x is a ByteTensor now
    return y
