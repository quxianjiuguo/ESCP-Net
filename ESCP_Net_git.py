from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from thop import profile
from timm.models.layers import trunc_normal_, DropPath
from torch import Tensor



class Encoder_stem(nn.Module):
    def __init__(self, in_ch,out_ch):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,1,bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, bias=False,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, bias=False, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.act =  nn.ReLU(inplace=True)
    def forward(self,x):
        x1 = self.conv3x3(x)
        x2 = self.conv1x1(x)

        return self.act(x1+x2)





class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)




class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block_nums=1, order=2):
        super().__init__()
        self.transition = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=1),
            # nn.MaxPool2d(2),

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.features = nn.ModuleList([
            Block(out_channels, order=order) for i in range(block_nums)])
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        x = self.transition(x)
        for blk in self.features:
            x = blk(x)
        x = self.downsample(x)

        return x

class ConvBn2d(nn.Module):
    def __init__(self, in_ch,ou_ch,kernel_size=1, padding=0):
        super(ConvBn2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch,ou_ch,kernel_size=kernel_size,padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(ou_ch)

    def forward(self,x):
        x = self.conv2d(x)
        x = self.bn(x)

        return x


class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_ch=out_channels, ou_ch=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        # print('spatial',x.size())
        x = F.sigmoid(x)
        return x


class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_ch=out_channels, ou_ch=int(out_channels / 2), kernel_size=1, padding=0)
        self.conv2 = ConvBn2d(in_ch=int(out_channels / 2), ou_ch=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = nn.AvgPool2d(x.size()[2:])(x)
        # print('channel',x.size())
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


import torch.nn as nn
import math


class Spatial_attention(nn.Module):
    def __init__(self, channels):
        super(Spatial_attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7,
                              padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


class CAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAM, self).__init__()
        self.channel = ECA_block(in_channels)
        self.space = Spatial_attention(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.channel(x)
        x = self.space(x)
        x = self.conv(x)

        return x


class AGs(nn.Module):
    def __init__(self, C):
        super(AGs, self).__init__()
        self.gate = nn.Linear(C, 3 * C)
        self.linear1 = nn.Linear(C, C)
        self.linear2 = nn.Linear(C, C)

    def forward(self, x1, x2):  # x1 高级    x2 低级
        B, C, H, W = x1.shape  # 记录高级特征的shape
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        gate = self.gate(x1).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        g1, g2, g3 = gate[0], gate[1], gate[2]
        x2 = torch.sigmoid(self.linear1(g1) + x2) * x2 + torch.sigmoid(g2) * torch.tanh(g3)
        x2 = self.linear2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)

        return torch.concat([x2, x1], dim=1)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.cam = CAM(in_channels, out_channels)
        self.AG = AGs(out_channels)

    def forward(self, x, e=None):
        x = self.cam(x)
        out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.AG(out, e)

        return x, out  # 先返回的辅助输出 再返回往下传的变量

class ESCP-Net(nn.Module):  # RepVGG
    def __init__(self,n_channels,n_classes):
        super(Hornet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.stage0 = Encoder_stem(1,32)
        self.stage1 = Down(32,64,1,2)   # 64  112 112
        self.stage2 = Down(64,128,2,2)  # 128 56 56
        self.stage3 = Down(128,256,3,2) # 256 28 28
        self.stage4 = Down(256,512,3,2) # 512 14 14

        self.decoder4 = Decoder(512,256)
        self.decoder3 = Decoder(512,128)
        self.decoder2 = Decoder(256,64)
        self.decoder1 = Decoder(128,32)
        # self.Decoder_stem = Decoder(32+64,64,64)

        self.seghead = nn.Conv2d(64+self.n_classes, n_classes, kernel_size=1)
        self.channels = [2**i for i in range(5,10)]  # 32 64 128 256 512
        self.sup_conv4 = nn.Sequential(
            nn.Conv2d( self.channels[3], n_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_classes)
        )
        self.sup_conv3 = nn.Sequential(
            nn.Conv2d(self.channels[2], n_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_classes)
        )
        self.sup_conv2 = nn.Sequential(
            nn.Conv2d(self.channels[1], n_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_classes)
        )
        self.sup_conv1 = nn.Sequential(
            nn.Conv2d(self.channels[0], n_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_classes)
        )
    def forward(self, x):


        x0 = self.stage0(x)  #   torch.Size([2, 32, 224, 224])
        x1 = self.stage1(x0)  #
        x2 = self.stage2(x1)  #
        x3 = self.stage3(x2)  #
        x4 = self.stage4(x3)  #

        x4_sup,x4_back = self.decoder4(x4,x3)        # torch.Size([4, 256, 14, 14])  torch.Size([4, 512, 28, 28])
        x3_sup,x3_back = self.decoder3(x4_back,x2)       # torch.Size([4, 128, 28, 28])  # torch.Size([2, 256, 56, 56])
        x2_sup,x2_back = self.decoder2(x3_back,x1)       # torch.Size([4, 64, 56, 56])   torch.Size([2, 128, 112, 112])
        x1_sup,x1_back = self.decoder1(x2_back,x0)       # torch.Size([4, 32, 112, 112]) # torch.Size([2, 64, 224, 224])

        x4_out = self.sup_conv4(x4_sup)
        x3_out = self.sup_conv3(x3_sup)
        x2_out = self.sup_conv2(x2_sup)
        x1_out = self.sup_conv1(x1_sup)


        x1_out =F.upsample(x1_out, size=x.shape[2:], mode='bilinear', align_corners=True)
        x2_out =F.upsample(x2_out, size=x.shape[2:], mode='bilinear', align_corners=True)
        x3_out = F.upsample(x3_out, size=x.shape[2:], mode='bilinear', align_corners=True)
        x4_out = F.upsample(x4_out, size=x.shape[2:], mode='bilinear', align_corners=True)


        total = self.seghead(torch.concat([(x1_out+x2_out+x3_out+x4_out),x1_back],dim=1))

        if self.training:
            # do not use torch.sigmoid for amp safe
            return  total,x1_out,x2_out,x3_out,x4_out   # 最后返回七个结果
        else:
            return total  # 这里有待商榷  对于多类+分割 究竟是torch.argmax()  还是torch.sigmoid？



if __name__ == '__main__':
    model = ESCP-Net(1,4)
    # # summary(model, input_size=(1, 224, 224))
    # tensor = torch.randn(2, 1, 224, 224)
    # print(model(tensor)[0].shape)
    # # print(model(tensor)[1].shape)
    # # print(model(tensor)[2].shape)
    # # print(model(tensor)[3].shape)
    # # print(model(tensor)[4].shape)
    # # print(model(tensor)[5].shape)
    # # print(model(tensor)[6].shape)
    # flops, params = profile(model, inputs=(torch.randn(2, 1, 224, 224),))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G') # 10.802190336G
    # print('Params = ' + str(params / 1000 ** 2) + 'M') # 3.50266M