import torch
from torch import nn
import torch.nn.functional as F

from module import VGG_Block, Residual_Block, RCAB

class Conv_Block(nn.Module):
    def __init__(self, in_channel:int, out_channel:int, mode:str='vgg'):
        """
        Convolution Block

        U-Netの中で使用する畳み込みブロック

        Args:
            in_channel (int): 入力チャンネル数
            out_channnel (int): 出力チャンネル数
            mode (str): 使用するモデル. 'vgg', 'resnet', 'rcab' から選択可能.
        """
        super(Conv_Block, self).__init__()
        if mode == 'vgg':
            self.Block = VGG_Block(in_channel, out_channel)
        elif mode == 'resnet':
            layer = [Residual_Block(in_channel, out_channel)] + [Residual_Block(out_channel, out_channel) for _ in range(3)]
            self.Block = nn.Sequential(*layer)
        elif mode == 'rcab':
            layer = [Residual_Block(in_channel, out_channel)] + [Residual_Block(out_channel, out_channel) for _ in range(2)] + [RCAB(out_channel, out_channel)]
            self.Block = nn.Sequential(*layer)
        else:
            assert False, f'{mode} mode is not found'

    def forward(self,x):
        out = self.Block(x)
        return out

class Down_Block(nn.Module):
    def __init__(self, in_channel:int, out_channnel:int, conv_mode:str='vgg', down_mode:str='max_pool'):
        """
        Down Block

        U-Netの中で使用するダウンサンプリングブロック

        Args:
            in_channel (int): 入力チャンネル数
            out_channnel (int): 出力チャンネル数
            conv_mode (str): 使用する畳み込みブロック. 'vgg', 'resnet', 'rcab' から選択可能.
            down_mode (str): 使用するプーリング層. 'max_pool', 'pixel_unshuffle' から選択可能.
        """
        super(Down_Block, self).__init__()
        self.downscale = 2
        if down_mode == 'max_pool':
            self.pool = nn.MaxPool2d(self.downscale)
            self.Block = Conv_Block(in_channel, out_channnel, mode=conv_mode)
        elif down_mode == 'pixel_unshuffle':
            self.pool = nn.PixelUnshuffle(self.downscale)
            self.Block = Conv_Block(in_channel*(self.downscale**2), out_channnel, mode=conv_mode)
        else:
            assert False, f'{down_mode} down_mode is not found'
    
    def forward(self,x):
        out = self.pool(x)
        out = self.Block(out)
        return out

class UP_Block(nn.Module):
    def __init__(self, in_channel, out_channnel, conv_mode:str='vgg', up_mode='upsample'):
        """
        UP Block

        U-Netの中で使用するアップサンプリングブロック

        Args:
            in_channel (int): 入力チャンネル数
            out_channnel (int): 出力チャンネル数
            conv_mode (str): 使用する畳み込みブロック. 'vgg', 'resnet', 'rcab' から選択可能.
            up_mode (str): 使用するアップサンプリング層. 'nearest', 'bilinear', 'bicubic', 'conv_transpose', 'pixel_shuffle' から選択可能.
        """
        super(UP_Block, self).__init__()
        if up_mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=up_mode)
        elif up_mode in ['bilinear', 'bicubic']:
            self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
        elif up_mode == 'conv_transpose':
            self.up = nn.ConvTranspose2d(in_channel//2, in_channel//2, 2, stride=2)
        elif up_mode == 'pixel_shuffle':
            self.up = nn.PixelShuffle(2)
        else:
            assert False, f'{up_mode} up_mode is not found'
        self.conv = Conv_Block(in_channel, out_channnel, mode=conv_mode)

    def forward(self, x1, x2):
        '''
        x1 : 後半側の入力(MaxPooling => UpSample or ConvTranspse の処理を行うのでこちらの方が画像サイズが小さくなる)
        x2 : 前半側のショートカットの入力
        '''
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY//2, diffX // 2, diffX - diffX//2))

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out

class Out_Block(nn.Module):
    def __init__(self, in_channnel, out_channel):
        """
        Output Block

        U-Netの中で使用する出力ブロック

        Args:
            in_channel (int): 入力チャンネル数
            out_channnel (int): 出力チャンネル数
        """
        super(Out_Block, self).__init__()
        self.conv = nn.Conv2d(in_channnel, out_channel, 1, bias=False)
        if out_channel == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)
        return out

class Variable_U_Net(nn.Module):
    def __init__(self, in_channel, out_classes, first_channnel=64, down_num=4, conv_mode='vgg', down_mode='max_pool', up_mode='nearest'):
        """
        Variable U-Net

        Args:
            in_channel (int): 入力チャンネル数
            out_channnel (int): 出力チャンネル数
            first_channnel (int): 最初の畳み込みブロックの出力チャンネル数
            down_num (int): ダウンサンプリングブロックの数
            conv_mode (str): 使用する畳み込みブロック. 'vgg', 'resnet', 'rcab' から選択可能.
            down_mode (str): 使用するプーリング層. 'max_pool', 'pixel_unshuffle' から選択可能.
            up_mode (str): 使用するアップサンプリング層. 'nearest', 'bilinear', 'bicubic', 'conv_transpose', 'pixel_shuffle' から選択可能.
        """
        super(Variable_U_Net,self).__init__()
        self.first_channnel = first_channnel
        self.down_num = down_num

        self.Conv1 = Conv_Block(in_channel, first_channnel, mode=conv_mode)
        self.down_block_list = []

        channnel_mag = 1
        for _ in range(self.down_num - 1):
            self.down_block_list.append(Down_Block(first_channnel*channnel_mag, first_channnel*channnel_mag*2, conv_mode=conv_mode, down_mode=down_mode))
            channnel_mag *= 2
        if up_mode == 'pixel_shuffle':
            self.down_block_list.append(Down_Block(first_channnel*channnel_mag, first_channnel*channnel_mag*4, conv_mode=conv_mode, down_mode=down_mode))
        else:
            self.down_block_list.append(Down_Block(first_channnel*channnel_mag, first_channnel*channnel_mag, conv_mode=conv_mode, down_mode=down_mode))
        channnel_mag *= 2
        self.up_block_list = []
        if up_mode == 'pixel_shuffle':
            for _ in range(self.down_num-1):
                self.up_block_list.append(UP_Block(first_channnel*channnel_mag, first_channnel*channnel_mag, conv_mode=conv_mode, up_mode=up_mode))
                channnel_mag //= 2
        else:
            for _ in range(self.down_num-1):
                self.up_block_list.append(UP_Block(first_channnel*channnel_mag, first_channnel*channnel_mag//4, conv_mode=conv_mode, up_mode=up_mode))
                channnel_mag //= 2
        self.up_block_list.append(UP_Block(first_channnel*2, first_channnel, conv_mode=conv_mode, up_mode=up_mode))
        self.down_block = nn.ModuleList(self.down_block_list)
        self.up_block = nn.ModuleList(self.up_block_list)
        self.Conv2 = Out_Block(first_channnel, out_classes)

    def forward(self, x):
        x_list = [self.Conv1(x)]
        for i in range(self.down_num):
            x_list.append(self.down_block[i](x_list[i]))
        out = x_list[self.down_num]
        
        for i in range(self.down_num):
            out = self.up_block[i](out, x_list[self.down_num-i-1])
        out = self.Conv2(out)
        return out
