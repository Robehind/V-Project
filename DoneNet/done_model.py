import torch.nn as nn


class DoneNet(nn.Module):
    def __init__(
        self,
        feat_sz,
        mid_sz,
        dprate
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_sz, mid_sz),
            nn.ReLU(),
            nn.Dropout(p=dprate),
            nn.Linear(mid_sz, 1),
            nn.Sigmoid())

    def forward(self, feat):
        return self.net(feat)
