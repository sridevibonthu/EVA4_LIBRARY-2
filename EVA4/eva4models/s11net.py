'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva4net import Net

class ResBlock(nn.Module):
    def __init__(self, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu1(self.bn2(self.conv2(out)))
        out = torch.add(x, out)
        return out

class S11Block(nn.Module):
    def __init__(self, in_planes, planes, parallel=True):
        super(S11Block, self).__init__()
        self.parallel = parallel
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        if parallel:
            self.res = ResBlock(planes)

    def forward(self, x):
        out = self.relu(self.bn(self.maxpool(self.conv(x))))
        if self.parallel:
            out = self.res(out)
        return out

#implementation of the new resnet model
class S11Net(Net):
  def __init__(self,name="S11Net", dropout_value=0):
    super(S11Net,self).__init__(name)
    self.prepLayer=self.create_conv2d(3, 64, dropout=dropout_value)

    self.layer1 = S11Block(64, 128)
    self.layer2 = S11Block(128, 256, False)
    self.layer3 = S11Block(256, 512)

    #ending layer or layer-4
    self.maxpool = nn.MaxPool2d(4)
    self.fc = self.create_conv2d(512, 10, kernel_size=(1,1), padding=0, bn=False, relu=False)


  def forward(self,x):
    x=self.prepLayer(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.maxpool(x)
    x = self.fc(x)
    x = x.view(-1,10)
    return F.log_softmax(x,dim=-1)


from eva4modeltrainer import ModelTrainer
from torchsummary import summary

class Net2(nn.Module):
    """
    Base network that defines helper functions, summary and mapping to device
    """
    def conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode)]

    def separable_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, padding_mode="zeros"):
      return [nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
              nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), bias=bias)]

    def activate(self, l, out_channels, bn=True, dropout=0, relu=True,max_pooling=0):
      if(max_pooling>0):
        l.append(nn.MaxPool2d(2,2))
      if bn:
        l.append(nn.BatchNorm2d(out_channels))
      if dropout>0:
        l.append(nn.Dropout(dropout))
      if relu:
        l.append(nn.ReLU())

      return nn.Sequential(*l)

    def create_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, groups=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros",max_pooling=0):
      return self.activate(self.conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, groups=groups, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode), out_channels, bn, dropout, relu,max_pooling)

    def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
      return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 out_channels, bn, dropout, relu)

    def __init__(self, name="Model"):
        super(Net2, self).__init__()
        self.trainer = None
        self.name = name

    def summary(self, input_size): #input_size=(1, 28, 28)
      summary(self, input_size=input_size)

    def gotrain(self, optimizer, train_loader, test_loader, epochs, statspath, scheduler=None, batch_scheduler=False, L1lambda=0):
      self.trainer = ModelTrainer(self, optimizer, train_loader, test_loader, statspath, scheduler, batch_scheduler, L1lambda)
      self.trainer.run(epochs)

    def stats(self):
      return self.trainer.stats if self.trainer else None

#implementation of the new resnet model
class newResnetS11(Net2):
  def __init__(self,name="Model",dropout_value=0):
    super(newResnetS11,self).__init__(name)
    self.prepLayer=self.create_conv2d(3, 64, dropout=dropout_value)
    #layer1
    self.layer1Conv1=self.create_conv2d(64,128, dropout=dropout_value,max_pooling=1)
    self.layer1resnetBlock1=self.resnetBlock(128,128)
    #layer2
    self.layer2Conv1=self.create_conv2d(128,256, dropout=dropout_value,max_pooling=1)
    #layer3
    self.layer3Conv1=self.create_conv2d(256,512, dropout=dropout_value,max_pooling=1)
    self.layer3resnetBlock1=self.resnetBlock(512,512)
    #ending layer or layer-4
    self.maxpool=nn.MaxPool2d(4)
    self.fc_layer=self.create_conv2d(512, 10, kernel_size=(1,1), padding=0, bn=False, relu=False)
  def resnetBlock(self,in_channels, out_channels):
      l=[]
      l.append(nn.Conv2d(in_channels,out_channels,(3,3),padding=1,bias=False))
      l.append(nn.BatchNorm2d(out_channels))
      l.append(nn.ReLU())
      l.append(nn.Conv2d(in_channels,out_channels,(3,3),padding=1,bias=False))
      l.append(nn.BatchNorm2d(out_channels))
      l.append(nn.ReLU())
      return nn.Sequential(*l)

  def forward(self,x):
    #prepLayer
    x=self.prepLayer(x)
    #Layer1
    x=self.layer1Conv1(x)
    r1=self.layer1resnetBlock1(x)
    x=torch.add(x,r1)
    #layer2
    x=self.layer2Conv1(x)
    #layer3
    x=self.layer3Conv1(x)
    r2=self.layer3resnetBlock1(x)
    x=torch.add(x,r2)
    #layer4 or ending layer
    x=self.maxpool(x)
    x=self.fc_layer(x)
    x=x.view(-1,10)
    return F.log_softmax(x,dim=-1)