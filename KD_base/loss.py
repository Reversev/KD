#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/12/29 15:37
# @Author : ''
# @FileName: loss.py
import torch
import torch.nn as nn


class ArcNet(nn.Module):
    def __init__(self, cls_num=10, feature_dim=512, m=1, s=2):
        super(ArcNet, self).__init__()
        self.w = nn.Parameter(torch.randn(feature_dim, cls_num))
        self.m = m
        self.s = s

    def forward(self, features):
        # 特征与权重 归一化
        _features = nn.functional.normalize(features, dim=1)
        _w = nn.functional.normalize(self.w, dim=0)

        # 特征向量与参数向量的夹角theta，分子numerator，分母denominator
        theta = torch.acos(torch.matmul(_features, _w) / 2)     # /2防止下溢
        numerator = torch.exp(self.s * torch.cos(theta + self.m))
        denominator = torch.sum(torch.exp(self.s * torch.cos(theta)), dim=1, keepdim=True) - torch.exp(
            self.s * torch.cos(theta)) + numerator
        return torch.log(torch.div(numerator, denominator))
