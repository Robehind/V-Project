import torch.nn as nn
import torchvision.models as tvmodels
import torch


class my_resnet50(nn.Module):
    def __init__(self):
        super(my_resnet50, self).__init__()
        resnet50 = tvmodels.resnet50(pretrained=True)
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
