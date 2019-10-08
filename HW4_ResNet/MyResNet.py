import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=1)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):

        residual = x


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class MyResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes, inplanes, width):
        
        super(MyResNet, self).__init__()
        
        self._norm_layer = nn.BatchNorm2d

        self._inplanes = inplanes
        self._dilation = 1

        self.conv1 = conv3x3(inplanes, 32, stride=1, padding=1)
        self._inplanes = 32
        self.bn1 = nn.BatchNorm2d(self._inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        self.conv2_x = self._make_layer(block, 32, layers[0], 1)
        self.conv3_x = self._make_layer(block, 64, layers[1], 2)
        self.conv4_x = self._make_layer(block, 128, layers[2], 2)
        self.conv5_x = self._make_layer(block, 256, layers[3], 2)
        
        self.mp = nn.MaxPool2d(kernel_size=2)
        
        last_width = int(width / (2**4))
        self.fc = nn.Linear(256*last_width*last_width, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        
        norm_layer = nn.BatchNorm2d
        downsample = None

        if stride != 1 or self._inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self._inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes=self._inplanes, planes=planes, stride=stride,
                            downsample=downsample))
        self._inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self._inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)


        out = self.mp(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out