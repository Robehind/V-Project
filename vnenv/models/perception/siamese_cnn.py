import torch
import torch.nn as nn


class SiameseLinear(nn.Module):
    def __init__(
        self,
        input_sz,
        output_sz
    ) -> None:
        super().__init__()
        self.shared_linear = nn.Linear(input_sz, output_sz)

    def forward(self, vobs, tobs):
        p1 = self.shared_linear(vobs)
        p2 = self.shared_linear(tobs)
        return torch.cat([p1, p2], dim=1)
