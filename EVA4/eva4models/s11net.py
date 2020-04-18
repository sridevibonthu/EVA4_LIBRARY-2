import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net

class ResBlock(nn.Module):
    def __init__(self, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        x += out
        return x

class S11Block(nn.Module):
    def __init__(self, in_planes, planes, parallel=True):
        super(S11Block, self).__init__()
        self.parallel = parallel
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        if parallel:
            self.res = ResBlock(planes)

    def forward(self, x):
        out = F.relu(self.bn(F.max_pool2d(self.conv(x), 2)))
        if self.parallel:
            out = self.res(out)
        return out

#implementation of the new resnet model
class S11Net(Net):
  def __init__(self,name="S11Net", dropout_value=0, num_classes=10):
    super(S11Net,self).__init__(name)
    self.prepLayer=self.create_conv2d(3, 64, dropout=dropout_value)
    self.num_classes = num_classes
    self.layer1 = S11Block(64, 128)
    self.layer2 = S11Block(128, 256, False)
    self.layer3 = S11Block(256, 512)

    #ending layer or layer-4
    self.fc = self.create_conv2d(512, num_classes, kernel_size=(1,1), padding=0, bn=False, relu=False)


  def forward(self,x):
    x=self.prepLayer(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = F.max_pool2d(x, x.size(-1))
    x = self.fc(x)
    x = x.view(-1,self.num_classes)
    return F.log_softmax(x,dim=-1)