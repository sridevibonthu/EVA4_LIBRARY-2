import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net

# class Encoder(nn.Module):
#     def __init__(self, inplanes, outplanes, dilation):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilation, stride=2, dilation=dilation, bias=False)
#         self.bn1 = nn.BatchNorm2d(outplanes)
#         self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
#         self.bn2 = nn.BatchNorm2d(outplanes)
#         self.xconv = nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, stride=2, bias=False)
#         self.bn = nn.BatchNorm2d(outplanes)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x))) # RF += j
#         out = self.bn2(self.conv2(out))
#         x = self.bn(self.xconv(x))
#         out = x + out
#         out = F.relu(out)
#         return out


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

class EncoderPath(nn.Module):
  def __init__(self, inplanes, outplanes, dilation):
      super(EncoderPath, self).__init__()
      self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=dilation, stride=2, dilation=dilation, bias=False)
      self.bn1 = nn.BatchNorm2d(outplanes)
      self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=dilation, stride=1, dilation=dilation, bias=False)
      self.bn2 = nn.BatchNorm2d(outplanes)

  def forward(self, x):
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
      return out


class EncoderBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(EncoderBlock, self).__init__()
        self.direct = nn.Conv2d(inplanes, outplanes//4, kernel_size=1, padding=0, stride=2, bias=False)
        self.directbn = nn.BatchNorm2d(outplanes//4)
        # if we need to reduce we can do groupwise here with shuffle, worth a try
        self.path1 = EncoderPath(inplanes, outplanes//4, 1)
        self.path2 = EncoderPath(inplanes, outplanes//4, 2)
        self.path3 = EncoderPath(inplanes, outplanes//4, 4)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        x = self.directbn(self.direct(x))
        out = torch.cat((x, p1, p2, p3), 1)
        out = F.relu(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, planes):
        super(DecoderBlock, self).__init__()
        #self.upsample = nn.ConvTranspose2d(planes*4, planes*4, kernel_size=3, stride=2, padding=1)
        # At this point we will use Pixel Shuffle to make resolution 224x224 
        planes = planes//4 #due to pixel shuffle
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False, groups = 4)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False, groups = 4)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.pixel_shuffle(x, 2) # 32 channels
        out = F.relu(self.bn1(self.conv1(out))) # 64 channels
        out = self.bn2(self.conv2(out)) # 128 channels
        return out

class Encoder(nn.Module):
  def __init__(self, planes):
    super(Encoder,self).__init__()
    self.encoder1 = EncoderBlock(planes, planes*2)   # 128 channels# RF = 24
    self.encoder2 = EncoderBlock(planes*2, planes*4)  # 256 channels
    self.encoder3 = EncoderBlock(planes*4, planes*8)  # 512 channels
   
  def forward(self,x):
    e1 = self.encoder1(x) # 32 channels 80x80
    e2 = self.encoder2(e1) # 64 channels 40x40
    e3 = self.encoder3(e2) # 128 channels 20x20

    return e1, e2, e3

class MinMaxScaler(nn.Module):
  def __init(self):
    super(MinMaxScaler, self).__init__()

  def forward(self, x):
    s = x.shape
    y = x.view(s[0], s[1], -1) 
    y = y - y.min(2, keepdim=True)[0]
    y = y/(y.max(2, keepdim=True)[0] )
    y = y.view(s)
    return y

class MaskDecoder(nn.Module):
  def __init__(self, planes):
    super(MaskDecoder,self).__init__()
    self.decoder1 = DecoderBlock(planes)   # 256 INPUT  AND 64 OUTPUT
    self.decoder2 = DecoderBlock(planes//2)  # 128 Input 32 output
    #e1 has 128 output
    self.e1conv = nn.Conv2d(planes//2, planes//4, kernel_size=3, padding=1, stride=1, bias=False)
    self.e1bn = nn.BatchNorm2d(planes//4) # 64 channels
    #e0 has 64 output
    self.e0conv = nn.Conv2d(planes//4, planes//8, kernel_size=3, padding=1, stride=1, bias=False)
    self.e0bn = nn.BatchNorm2d(planes//8) # 32 channels

    self.conv1 = nn.Conv2d(planes//4, planes//8, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes//8)
    
    self.conv2 = nn.Conv2d(planes//8, planes//8, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes//8)

    self.conv3 = nn.Conv2d(planes//8, 1, kernel_size=1, stride=1, bias=False)
    self.minmaxscaler = MinMaxScaler()

   
  def forward(self, *inputs):
    x, e1, e0 = inputs
    mask = self.decoder1(x) # 32 channels 80x80
    e1 = self.e1bn(self.e1conv(e1))
    mask = F.relu(torch.cat((mask, e1), 1))
    
    mask = self.decoder2(mask) # 64 channels 40x40
    e0 = self.e0bn(self.e0conv(e0))
    mask = F.relu(torch.cat((mask, e0), 1))
    
    mask = F.relu(self.bn1(self.conv1(mask)))
    mask = F.relu(self.bn2(self.conv2(mask)))
    
    mask = self.conv3(mask)
    # TODO: min-max scaling
    return self.minmaxscaler(mask)

class DepthDecoder(nn.Module):
  def __init__(self, planes):
    super(DepthDecoder,self).__init__()
    self.decoder1 = DecoderBlock(planes)   # 512 INPUT  AND 128 OUTPUT
    self.decoder2 = DecoderBlock(planes//2)  # 256 Input 64 output
    self.decoder3 = DecoderBlock(planes//4)  # 128 Input 32 output
    # e2 has 256 outputs
    self.e2conv = nn.Conv2d(planes//2, planes//4, kernel_size=3, padding=1, stride=1, bias=False)
    self.e2bn = nn.BatchNorm2d(planes//4) # 64 channels
    #e1 has 128 outputs
    self.e1conv = nn.Conv2d(planes//4, planes//8, kernel_size=3, padding=1, stride=1, bias=False)
    self.e1bn = nn.BatchNorm2d(planes//8) # 64 channels
    # e0 has 64 outputs
    self.e0conv = nn.Conv2d(planes//8, planes//16, kernel_size=3, padding=1, stride=1, bias=False)
    self.e0bn = nn.BatchNorm2d(planes//16) # 32 channels

    self.conv1 = nn.Conv2d(planes//8, planes//8, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes//8)
    
    self.conv2 = nn.Conv2d(planes//8, planes//8, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes//8)

    self.conv3 = nn.Conv2d(planes//8, 1, kernel_size=1, stride=1, bias=False)
    self.minmaxscaler = MinMaxScaler()

   
  def forward(self, *inputs):

    x, e2, e1, e0 = inputs
    depth = self.decoder1(x) # 32 channels 80x80
    e2 = self.e2bn(self.e2conv(e2))
    depth = F.relu(torch.cat((depth, e2), 1))

    depth = self.decoder2(x) # 32 channels 80x80
    e1 = self.e1bn(self.e1conv(e1))
    depth = F.relu(torch.cat((depth, e1), 1))
    
    depth = self.decoder3(depth) # 64 channels 40x40
    e0 = self.e0bn(self.e0conv(e0))
    depth = F.relu(torch.cat((depth, e0), 1))
    
    depth = F.relu(self.bn1(self.conv1(depth)))
    depth = F.relu(self.bn2(self.conv2(depth)))
    
    depth = self.conv3(depth)
    return self.minmaxscaler(depth)


#implementation of the new resnet model
class S15NetED(Net):
  def __init__(self,name="S15NetEncoderDecoder", planes=16):
    super(S15NetED,self).__init__(name)
    self.prepLayer = InitialBlock(planes)  # 64 channels
    self.encoder = Encoder(planes*4)  # 512 channels
    self.maskdecoder = MaskDecoder(planes*16)
    self.depthdecoder = DepthDecoder(planes*32)
   
  def forward(self,x):
    data_shape = x.size()
    e0 = self.prepLayer(x) # 32 channels 160x160
    e1, e2, e3 = self.encoder(e0) # 32 channels 80x80

    mask = self.maskdecoder(e2, e1, e0)
    depth = self.depthdecoder(e3, e2, e1, e0)
    
    return torch.cat((mask, depth), 1)
    
