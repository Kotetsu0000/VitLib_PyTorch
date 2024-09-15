import torch
from torch import nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channnel):
        super(Conv_Block, self).__init__()
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

class Down_Block(nn.Module):
    def __init__(self, in_channel, out_channnel):
        super(Down_Block, self).__init__()
        self.Mpool = nn.MaxPool2d(2)
        self.Block = Conv_Block(in_channel, out_channnel)
    
    def forward(self,x):
        out = self.Mpool(x)
        out = self.Block(out)
        return out

class UP_Block(nn.Module):
    def __init__(self, in_channel, out_channnel, bilinear=True):
        super(UP_Block, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channel//2, in_channel//2, 2, stride=2)
        self.conv = Conv_Block(in_channel, out_channnel)

    def forward(self, x1, x2):
        '''
        x1 : 後半側の入力(MaxPooling => UpSample or ConvTranspse の処理を行うのでこちらの方が画像サイズが小さくなる)
        x2 : 前半側のショートカットの入力
        '''
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffY // 2, diffY - diffY//2, diffX // 2, diffX - diffX//2))
        try:
            out = torch.cat([x2, x1], dim=1)
        except:
            print(x1.size(), x2.size())
            exit()
        out = self.conv(out)
        return out

class Out_Block(nn.Module):
    def __init__(self, in_channnel, out_channel):
        super(Out_Block, self).__init__()
        self.conv = nn.Conv2d(in_channnel, out_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)
        return out

class U_Net(nn.Module):
    def __init__(self, in_channel, out_classes, bilinear=True):
        super(U_Net,self).__init__()
        self.Conv1 = Conv_Block(in_channel, 64) #128x128xin_channel => #128x128x64
        self.Block1 = Down_Block(64, 128)       # => #64x64x128
        self.Block2 = Down_Block(128, 256)      # => #32x32x256
        self.Block3 = Down_Block(256, 512)      # => #16x16x512
        self.Block4 = Down_Block(512, 512)      # => #8x8x512
        self.Block5 = UP_Block(1024, 256, bilinear=bilinear)
        self.Block6 = UP_Block(512, 128, bilinear=bilinear)
        self.Block7 = UP_Block(256, 64, bilinear=bilinear)
        self.Block8 = UP_Block(128, 64, bilinear=bilinear)
        self.Conv2 = Out_Block(64,out_classes)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Block1(x1)
        x3 = self.Block2(x2)
        x4 = self.Block3(x3)
        out = self.Block4(x4)
        
        out = self.Block5(out, x4)
        out = self.Block6(out, x3)
        out = self.Block7(out, x2)
        out = self.Block8(out, x1)
        out = self.Conv2(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            #*discriminator_block(256, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
        #return self.model(img_A)
