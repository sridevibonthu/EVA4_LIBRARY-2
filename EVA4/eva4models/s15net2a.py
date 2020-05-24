import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net

class Encoder(nn.Module):
    def __init__(self, inplanes, outplanes, dilation):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=dilation, stride=2, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.xconv = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        x = self.bn(self.xconv(x))
        out = x + out
        out = F.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, planes):
        super(Decoder, self).__init__()
        #self.upsample = nn.ConvTranspose2d(planes*4, planes*4, kernel_size=3, stride=2, padding=1)
        # At this point we will use Pixel Shuffle to make resolution 224x224 
        planes = planes//4 #due to pixel shuffle
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*2)

    def forward(self, x):
        out = F.pixel_shuffle(x, 2) # 32 channels
        out = F.relu(self.bn1(self.conv1(out))) # 64 channels
        out = F.relu(self.bn2(self.conv2(out))) # 128 channels
        return out
        

class InitialBlock(nn.Module):
    def __init__(self, planes):
        super(InitialBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*2)
        #self.conv3 = nn.Conv2d(planes*2, planes*4, kernel_size=3, padding=1, stride=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(planes*4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        #out = F.relu(self.bn3(self.conv3(out)))
        return  out

#implementation of the new resnet model
class S15Net2a(Net):
  def __init__(self,name="S15Net2a", outchannels=2, planes=16):
    super(S15Net2a,self).__init__(name)
    self.prepLayer = InitialBlock(planes)  # IN: 160x160x3, OUT 80x80x128, JUMP = 2, RF = 7
    self.encoder1 = Encoder(planes*2, planes*2, 2)   # RF = 24
    self.encoder2 = Encoder(planes*2, planes*4, 2)   # RF = 48
    self.encoder3 = Encoder(planes*4, planes*8, 2)   # RF = 80
    
    self.decoder1 = Decoder(planes*8)   # RF = 24
    self.decoder2 = Decoder(planes*8)   # RF = 48
    
    dplanes = planes*2  + planes * 4
    self.decoder3 = Decoder(dplanes)   # RF = 80
    planes = dplanes // 2 + planes * 2

    self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(planes, outchannels, kernel_size=3, padding=1, stride=1, bias=False)
   
  def forward(self,x):
    data_shape = x.size()
    x = self.prepLayer(x) # 32 channels 160x160
    e1 = self.encoder1(x) # 32 channels 80x80
    e2 = self.encoder2(e1) # 64 channels 40x40
    e3 = self.encoder3(e2) # 128 channels 20x20

    d1 = torch.cat((e2, self.decoder1(e3)), 1) # 128 channels 40x40
    d2 = torch.cat((e1, self.decoder2(d1)), 1) # 96 channels 80x80
    d3 = torch.cat((x, self.decoder3(d2)), 1) # 80 channels 160x160
  
  
    out = F.relu(self.bn1(self.conv1(d3)))
    out = self.conv2(out)
    outshape = out.size()
    
    # min max scaling
    y = out.view(outshape[0], outshape[1], -1) 
    y = y - y.min(2, keepdim=True)[0]
    y = y/(y.max(2, keepdim=True)[0] )
    y = y.view(outshape)
    return y
