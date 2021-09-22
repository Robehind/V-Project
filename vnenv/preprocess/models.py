import torch
import torch.nn as nn
import torchvision.models as models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def conv2dout_sz(H, K, S, P):
    return (H + 2*P - K)//S + 1


def deconv2dout_sz(H, K, S, P):
    return (H-1)*S-2*P+(K-1)+1


def CNNout_sz(net, h, w):
    H, W, out_c = CNNout_HWC(net, h, w)
    return H*W*out_c


def CNNout_HWC(net, h, w):
    H, W = h, w
    for c in net:
        name = c.__class__.__name__
        if 'Conv2d' in name:
            out_c = c.out_channels
            k, s, p = c.kernel_size, c.stride, c.padding
            H = conv2dout_sz(H, k[0], s[0], p[0])
            W = conv2dout_sz(W, k[1], s[1], p[1])
        elif 'Pool' in name:
            k, s, p = c.kernel_size, c.stride, c.padding
            H = conv2dout_sz(H, k, s, p)
            W = conv2dout_sz(W, k, s, p)
        elif 'sample' in name:
            a, b = c.scale_factor
            H *= a
            W *= b
        elif 'ConvTranspose2d' in name:
            out_c = c.out_channels
            k, s, p = c.kernel_size, c.stride, c.padding
            H = deconv2dout_sz(H, k[0], s[0], p[0])
            W = deconv2dout_sz(W, k[1], s[1], p[1])

    return int(H), int(W), int(out_c)


class my_resnet50(nn.Module):
    def __init__(self):
        super(my_resnet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.eval()

        resnet50_fc = list(resnet50.children())[:-1]
        self.resnet50_fc = nn.Sequential(*resnet50_fc)
        self.resnet50_fc.eval()

        resnet50_s = list(resnet50.children())[-1:]
        self.resnet50_s = nn.Sequential(*resnet50_s)
        self.resnet50_s.eval()

    def forward(self, x):
        with torch.no_grad():
            resnet_fc = self.resnet50_fc(x).squeeze()
            resnet_s = self.resnet50_s(resnet_fc).squeeze()
        return dict(fc=resnet_fc, s=resnet_s)


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, 4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            # nn.Flatten()
            )

    def forward(self, input_):
        return self.net(input_)


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.net = nn.Sequential(
            # 128 4 4
            nn.Upsample(scale_factor=(2, 2), mode='bilinear',
                        align_corners=True),
            # 128 8 8
            nn.ConvTranspose2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            # 128 8 8
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),
            # 128 16 16
            nn.ConvTranspose2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            # 64 16 16
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),
            # 64 32 32
            nn.ConvTranspose2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            # 32 32 32
            nn.ConvTranspose2d(32, 3, 8, 4, padding=2),
            nn.ReLU(inplace=True),
            )

    def forward(self, input_):
        return self.net(input_)


class Decoder2(nn.Module):
    def __init__(
        self,
        input_channels=128
    ):
        super(Decoder2, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, padding=1),
            )

    def forward(self, input_):
        return self.net(input_)


class EncoderT(nn.Module):
    def __init__(self):
        super(EncoderT, self).__init__()
        ndf = 64
        self.net = nn.Sequential(
            # 3 128 128
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 64 64
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 32 32
            nn.Conv2d(ndf*2, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 16 16
            nn.MaxPool2d(2, 2),
            # 128 8 8
            nn.Conv2d(ndf*2, ndf*2, 5, 1, 0, bias=False),
            nn.Sigmoid(),
            # 128 4 4
        )
        self.apply(weights_init)

    def forward(self, input_):
        return self.net(input_)


class DecoderT(nn.Module):
    '''pytorch tutorialsGAN'''
    def __init__(
        self,
        input_channels=128
    ):
        super(DecoderT, self).__init__()
        self.input_channels = input_channels
        ngf = 64
        self.net = nn.Sequential(
            # 128 4 4
            nn.ConvTranspose2d(input_channels, ngf*2, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # 128 8 8
            nn.Upsample(scale_factor=(2, 2), smode='nearest'),
            # 128 16 16
            nn.ConvTranspose2d(ngf*2, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # 128 32 32
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            # 64 64 64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 3 128 128
            )
        self.apply(weights_init)

    def forward(self, input_):
        return self.net(input_)
