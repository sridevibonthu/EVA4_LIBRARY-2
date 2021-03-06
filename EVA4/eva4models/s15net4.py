import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = x + out
        out = F.relu(out)
        return out

class InitialBlock(nn.Module):
    def __init__(self, planes):
        super(InitialBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return  out

#implementation of the new resnet model
class S15Net4(Net):
  def __init__(self,name="S15Net4", outchannels=256, planes=32):
    super(S15Net4,self).__init__(name)
    self.prepLayer = InitialBlock(planes)               # IN: 160x160x3, OUT 80x80x128, JUMP = 2, RF = 7
    self.layer1 = ResBlock(planes*4, planes, 2)   # RF = 24
    self.layer2 = ResBlock(planes*4, planes, 3)   # RF = 48
    self.layer3 = ResBlock(planes*4, planes, 4)   # RF = 80
    #self.layer4 = ResBlock(planes*4, planes, 16)  # RF = 248

    self.upsample = self.create_conv2d(planes*4, planes*16, kernel_size=(1,1), padding=0) # IN 80x80x128, OUT 80x80x512, RF = 120 
    #self.upsample = nn.ConvTranspose2d(planes*4, planes*4, kernel_size=3, stride=2, padding=1)
    # At this point we will use Pixel Shuffle to make resolution 224x224 
    self.conv1 = nn.Conv2d(planes*4, planes*4, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes*4)
    self.conv2 = nn.Conv2d(planes*4, planes*8, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes*8)

    # we will quantize depth in 16 classes.
    self.conv3 = nn.Conv2d(planes*8, outchannels, kernel_size=1, padding=0, stride=1, bias=False)
   
  def forward(self,x):
    data_shape = x.size()
    x = self.prepLayer(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    #x = self.layer4(x)
  
    #out = self.upsample(x, output_size=data_shape)
    out = self.upsample(x)
    out = F.pixel_shuffle(out, 2)
    # rather than probabilities we are making it a hard mask prediction
    # this is not it, we can restore binary logic later

    out = F.relu(self.bn1(self.conv1(out)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.conv3(out)
    

    # we can do away with this and use sigmoid
    # but its better to use softmax with cross entropy in case of depth
    # while we can try bcewith logits for mask

    # now we need to apply softmax on all from 1:
    # b, 256, h, w
    depth = F.softmax(out[:, 1:, :, :], 1)
    # apply softmax on depth

    mask = out[:, :1, :, :]
    outshape = mask.size()
    # do minmax scaling for mask
    mask = mask.view(outshape[0], outshape[1], -1) 
    mask = mask - mask.min(2, keepdim=True)[0]
    mask = mask/(mask.max(2, keepdim=True)[0] )
    mask = mask.view(outshape)

    return torch.cat([mask, depth], 1)
    

    # now we can use DICE loss, BCE loss, BCE loss with logits
    # the custom loss will have to apply onehot encoding to the depth image
    # the depth image  generator have to reverse create image from oneho encoded data
    # That image need to be displayed in tensorboard
    # this has to be done without affecting the current working
