import torch.nn as nn
import torch
import einops


class VTencoder(nn.Module):
    def __init__(
        self,
        feat_dim=512,
        n_head=8,
        hid_dim=512,
        dropout=0
    ):
        super(VTencoder, self).__init__()
        self.attention = nn.MultiheadAttention(
            feat_dim, n_head, dropout=dropout
        )
        self.linear = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, feat_dim),
            nn.Dropout(p=dropout)
        )
        self.layernorm = nn.LayerNorm(feat_dim, eps=1e-6)

    def forward(self, x: torch.Tensor):
        # input  patches * batch * feat_dim
        res = x
        out, _ = self.attention.forward(x, x, x)
        out = self.linear(out) + res
        # output  patches * batch * feat_dim
        return self.layernorm(out)


class ViT(nn.Module):
    def __init__(
        self,
        patches,
        feat_dim,
        n_head=8,
        n_encoder=6,
    ):
        super(ViT, self).__init__()
        self.encoders = nn.ModuleList([
            VTencoder(feat_dim, n_head) for _ in range(n_encoder)
        ])
        self.pos_embedding = nn.Parameter(torch.randn(1, patches+1, feat_dim))
        self.token = nn.Parameter(torch.randn(1, 1, feat_dim))

    def forward(self, x):
        # input batch * patches * feat_dim
        # cat token
        b_sz = x.shape[0]
        tokens = einops.repeat(self.token, f'b p d -> ({b_sz} b) p d')
        x = torch.cat((x, tokens), dim=1)
        # add pos embeddings
        x += self.pos_embedding
        x = einops.rearrange(x, 'b p d -> p b d')
        for encoder in self.encoders:
            x = encoder(x)
        x = einops.rearrange(x, 'p b d -> b p d')
        # output batch * feat_dim
        return x[:, 0]


if __name__ == '__main__':
    net = ViT(12, 512)
    inp = torch.randn(32, 12, 512)
    a = net(inp)
    print(a.shape)
