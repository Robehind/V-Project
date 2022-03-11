from typing import Dict
import torch.nn as nn
import torchvision.models as tvmodels
import torch
import numpy as np


class resnet18fm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        res18 = tvmodels.resnet18(pretrained=True)
        Tres18 = list(res18.children())[:-2]
        self.Tres18 = nn.Sequential(*Tres18)
        self.Tres18.eval()

    def forward(self, x) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            fm = self.Tres18(x).squeeze()
        return dict(resnet18fm=fm.cpu().numpy())


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

    def forward(self, x) -> Dict[str, np.ndarray]:
        with torch.no_grad():
            resnet_fc = self.resnet50_fc(x).squeeze()
            resnet_s = self.resnet50_s(resnet_fc).squeeze()
        return dict(resnet50fc=resnet_fc.cpu().numpy(),
                    resnet50score=resnet_s.cpu().numpy())
