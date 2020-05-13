import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        x += out
        return x

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
        return  F.max_pool2d(out, 2)

#implementation of the new resnet model
class S15Net(Net):
  def __init__(self,name="S15Net", planes=32):
    super(S15Net,self).__init__(name)
    self.prepLayer = InitialBlock()               # IN: 160x160x3, OUT 80x80x128, JUMP = 2, RF = 7
    self.layer1 = ResBlock(planes*4, planes, 2)   # RF = 24
    self.layer2 = ResBlock(planes*4, planes, 4)   # RF = 56
    self.layer3 = ResBlock(planes*4, planes, 8)   # RF = 120
    #self.layer4 = ResBlock(planes*4, planes, 16)  # RF = 248

    self.upres_conv = self.create_conv2d(planes*4, planes*16, kernel_size=(1,1), padding=0) # IN 80x80x128, OUT 80x80x512, RF = 120 

    # At this point we will use Pixel Shuffle to make resolution 224x224 
    self.mask_conv1 = self.create_conv2d(planes*4, planes*4) # IN 160x160x128, OUT 224x224x128, RF = 250
    self.mask_conv2 = self.create_conv2d(planes*4, planes*8) # IN 224x224x128, OUT 224x224x256, RF = 252
    self.depth_conv1 = self.create_conv2d(planes*4, planes*4) # IN 224x224x128, OUT 224x224x128, RF = 250
    self.depth_conv2 = self.create_conv2d(planes*4, planes*8) # IN 224x224x128, OUT 224x224x256, RF = 252

    self.mask_out = self.create_conv2d(planes*8, 1, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN 224x224x256, OUT 224x224x1, RF = 252 
    self.depth_out = self.create_conv2d(planes*8, 1, kernel_size=(1,1), padding=0, bn=False, relu=False) # IN 224x224x256, OUT 224x224x1, RF = 252 

  def forward(self,x):
    x=self.prepLayer(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    #x = self.layer4(x)
    
    x = F.pixel_shuffle(self.upres_conv(x), 2)

    # rather than probabilities we are making it a hard mask prediction
    mask = F.sigmoid(self.mask_out(self.mask_conv2(self.mask_conv1(x)))) > 0.5
    mask = mask.float() # cast back to float sicne x is a ByteTensor now
    
    depth = F.sigmoid(self.depth_out(self.depth_conv2(self.depth_conv1(x))))
    # we should be applying sigmoid activation on these and for mask we can even apply threshold of 0.5 to give binary image
    return torch.stack(mask, depth)