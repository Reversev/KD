#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/12/23 21:00
# @Author : ''
# @FileName: model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# build for 18 and 34 layers
class BasicBlock(nn.Module):
    expansion = 1  # 18 and 34 set 1 (first layer * 1 -> last layer) , others set 4 (first layer * 4 -> last layer) in these block

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        # stride = 1 correspond to shortcut but stride set other values correspond to shortcut with conv1x1
        # if stride == 1, output = (input - 3 + 2 * 1)/1 + 1 = input
        # if stride == 2, output = (input - 3 + 2 * 1)/2 + 1 = input/2 + 0.5 = input/2 (low)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# build for 50, 101 and 152 layers
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)   # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=3, stride=1, bias=False)   # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        # block == BasicBlock or Bottleneck
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        # modify conv7 -> conv3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        shortcut = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, shortcut=shortcut, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            fea = torch.flatten(x, 1)  # fea: batch_size, 512
            # print(fea.shape)
            x = self.fc(fea)
        return fea, x


class _Convlayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate):
        super(_Convlayer, self).__init__()

        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('relu', nn.ReLU(inplace=True)),
        self.add_module('norm', nn.BatchNorm2d(num_output_features)),

        self.drop_rate = drop_rate

    def forward(self, x):
        x = super(_Convlayer, self).forward(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


