#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ネットワークを構成するモジュール"""
import warnings

from torch import nn
import torch.nn.functional as F

class VGG_Block(nn.Module):
    """VGG Block

    VGGブロック(https://arxiv.org/abs/1409.1556)の実装

    Args:
        in_channel (int): 入力チャンネル数
        out_channnel (int): 出力チャンネル数    
    """
    def __init__(self, in_channel:int, out_channnel:int):
        super(VGG_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channnel, 3, padding = 1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_channnel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channnel, out_channnel, 3, padding = 1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(out_channnel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out
    
class Residual_Block(nn.Module):
    """Residual Block

    Residualブロック(https://arxiv.org/abs/1512.03385)の実装

    Args:
        in_channel (int): 入力チャンネル数
        out_channnel (int): 出力チャンネル数
    """
    def __init__(self, in_channel:int, out_channnel:int):
        super(Residual_Block, self).__init__()
        if out_channnel % 4 != 0:
            warnings.warn('out_channnel % 4 != 0', UserWarning)

        self.conv1 = nn.Conv2d(in_channel, out_channnel//4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channnel//4)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channnel//4, out_channnel//4, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(out_channnel//4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channnel//4, out_channnel, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channnel)
        
        if in_channel != out_channnel:
            self.identity = nn.Conv2d(in_channel, out_channnel, kernel_size=1)
        else:
            self.identity = None
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.identity is not None:
            x = self.identity(x)
        out += x
        out = self.relu(out)
        return out

class CALayer(nn.Module):
    """Channel Attention (CA) Layer
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block

    Residual Channel Attention Block(https://arxiv.org/abs/1807.02734)の実装

    Args:
        in_channel (int): 入力チャンネル数
        out_channnel (int): 出力チャンネル数
        reduction (int): 中間層でのチャンネル数削減率
        bn (bool): Batch Normalizationを使用するかどうか
    """
    def __init__(self, in_channel:int, out_channnel:int, reduction:int=16, bn:bool=False):
        super(RCAB, self).__init__()
        if out_channnel < reduction:
            warnings.warn(f'out_channnel < reduction. so reducation change {out_channnel}', UserWarning)
            reduction = out_channnel
        module_list = [nn.Conv2d(in_channel, out_channnel, 3, padding = 1, padding_mode='reflect')]
        if bn: module_list.append(nn.BatchNorm2d(out_channnel))
        module_list.append(nn.ReLU(inplace=True))
        module_list.append(nn.Conv2d(out_channnel, out_channnel, 3, padding = 1, padding_mode='reflect'))
        module_list.append(nn.ReLU(inplace=True))
        module_list.append(CALayer(out_channnel, reduction))
        self.module = nn.Sequential(*module_list)

        if in_channel != out_channnel:
            self.identity = nn.Conv2d(in_channel, out_channnel, kernel_size=1)
        else:
            self.identity = None

    def forward(self,x):
        out = self.module(x)
        if self.identity is not None:
            x = self.identity(x)
        out += x
        return out

def calc_sum_params(model: nn.Module, verbose: bool=False) -> tuple:
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()

    G = 1000**3
    M = 1000**2
    K = 1000
    if params > G:
        if verbose:print(f"{params/G:.2f}G")
        params_text = f"{params/G:.2f}G"
    elif params > M:
        if verbose:print(f"{params/M:.2f}M")
        params_text = f"{params/M:.2f}M"
    elif params > K:
        if verbose:print(f"{params/K:.2f}K")
        params_text = f"{params/K:.2f}K"
    else:
        if verbose:print(f"{params}")
        params_text = f"{params}"
    return params_text, params
