# coding: utf-8
import math
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.gloablavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.layer4(x)
        x = self.gloablavgpool(x1)
        x= x.view(x.size(0),-1)
        x = self.bn(x)
        return x, x1


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        # self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        self.gru.flatten_parameters()

        out, _ = self.gru(x)
        out = out.mean(1)
        # out = self.dropout(out)
        out = self.fc(out)  # predictions based on every time step otherwise out[:,-1,:] main.py mean delete

        return out


class Lipreading(nn.Module):
    def __init__(self, mode, inputDim=512, hiddenDim=1024, nClasses=500, frameLen=29, every_frame=True):
        super(Lipreading, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.every_frame = every_frame
        self.nLayers = 3
        # frontend3D
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        # resnet

        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.inputDim)
        self.dropout = nn.Dropout(0.5)



        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame)
            # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x, x1 = self.resnet18(x)

        x = self.dropout(x)

        x = x.view(-1, self.frameLen, self.inputDim)
        x = self.gru(x)

        return x, x1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.l0 = nn.Linear(512 + 500, 512)
        self.l1 = nn.Linear(512, 1)

    def forward(self, y, M):
        # # [B,64,22,22]
        # h = F.relu(self.c0(M))
        # # [B,64,20,20]
        # h = self.c1(h)
        # # [B,32,18,18]h
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = torch.sigmoid(self.l1(h))
        #return [B,1]
        return h
class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(512+500, 256, kernel_size=1)
        self.c1 = nn.Conv2d(256, 1, kernel_size=1)
        # self.c2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        # h = F.relu(self.c1(h))
        # retunr [B,1,22,22]
        return torch.sigmoid(self.c1(h))
if (__name__ == '__main__'):
    import torchsummary
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=500)
    torchsummary.summary(model,(64,22,22),device='cpu')

