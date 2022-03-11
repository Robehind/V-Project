import torch
import torch.nn as nn
import torch.nn.functional as F
#辅助任务
class TTRGBpred(nn.Module):
    '''pytorch tutorialsGAN'''
    def __init__(
        self,
        input_channels = 128
    ):
        super(TTRGBpred, self).__init__()
        self.input_channels = input_channels
        ngf = 64
        self.net = nn.Sequential(
            # 128 4 4
            nn.ConvTranspose2d(input_channels, ngf*2, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #128 8 8
            nn.Upsample(scale_factor=(2,2),mode='nearest'),
            #128 16 16
            nn.ConvTranspose2d(ngf*2, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #128 32 32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            #64 64 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            #3 128 128
            )

    def forward(self, input_):
        return self.net(input_)

class SplitRGBPred(nn.Module):
    def __init__(self):
        super(SplitRGBPred, self).__init__()
        self.net = nn.Sequential(
            #128 4 4
            nn.Upsample(scale_factor=(2,2),mode='bilinear', align_corners=True),
            #128 8 8
            nn.ConvTranspose2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            #128 8 8
            nn.Upsample(scale_factor=(2,2),mode='nearest'),
            #128 16 16
            nn.ConvTranspose2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            #64 16 16
            nn.Upsample(scale_factor=(2,2),mode='nearest'),
            #64 32 32
            nn.ConvTranspose2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            #32 32 32
            nn.ConvTranspose2d(32, 3, 8, 4, padding =2),
            nn.ReLU(inplace=True),
            )
    def forward(self, input_):
        return self.net(input_)


class RGBpred(nn.Module):
#参考自splitNet的RGB还原网络
    def __init__(
        self,
        input_channels = 128
    ):
        super(RGBpred, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, padding=1),
            )
        

    def forward(self, x, params = None):

        if params == None:
            out = self.net(x)
        else:
            raise NotImplementedError
        return out

    #def output_sz(self,h,w):
        #return CNNout_sz(self.net,h,w)