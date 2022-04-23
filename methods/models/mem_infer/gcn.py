import torch.nn as nn
import torch
import h5py
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


class YangGCN(nn.Module):
    def __init__(
        self,
        adj_path: str = "../vdata/gcn/adjmat.dat",
        obj_path: str = "../vdata/gcn/objects.txt",
        wd_path: str = "../vdata/word_embedding/word_embedding.hdf5",
        wd_type: str = "fasttext",
        wd_sz: int = 300,
        input_sz: int = 1000,
        output_sz: int = 512,
        gcn_in_sz: int = 1024,
        gcn_hid_sz: int = 1024
    ):
        super(YangGCN, self).__init__()

        adj = torch.load(adj_path)
        adj = stdlize_adj(adj)
        self.register_buffer('adj', norm_adj(adj))
        with open(obj_path, 'r') as f:
            objects = f.readlines()
        objects = [o.strip() for o in objects]
        self.obj_num = len(objects)
        assert self.obj_num == self.adj.shape[0]

        # 构造词嵌入特征矩阵
        self.register_buffer('wd_clues', torch.zeros(self.obj_num, 300))
        wdf = h5py.File(wd_path, "r")
        wd = wdf[wd_type]
        for i in range(self.obj_num):
            self.wd_clues[i, :] = torch.from_numpy(wd[objects[i]][:])
        wdf.close()

        map2sz = gcn_in_sz // 2
        self.wd_linear = nn.Linear(wd_sz, map2sz)
        self.input_linear = nn.Linear(input_sz, map2sz)

        # GCN net
        self.gc1 = GraphConv(map2sz*2, gcn_hid_sz)
        self.gc2 = GraphConv(gcn_hid_sz, gcn_hid_sz)
        self.gc3 = GraphConv(gcn_hid_sz, 1)

        self.mapping = nn.Linear(self.obj_num, output_sz)

    def forward(self, x):
        input_embed = self.input_linear(x)
        word_embed = self.wd_linear(self.wd_clues)
        batch_sz = input_embed.shape[0]
        input_embed = input_embed.repeat(self.obj_num, 1, 1)
        x = torch.cat(
            (input_embed.permute(1, 0, 2),
             word_embed.repeat(batch_sz, 1, 1)),
            dim=2)
        x1 = F.relu(self.gc1(x, self.adj))
        x2 = F.relu(self.gc2(x1, self.adj))
        x3 = F.relu(self.gc3(x2, self.adj))
        x4 = x3.squeeze(-1)
        x5 = self.mapping(x4)
        return x5


# Code borrowed from https://github.com/tkipf/pygcn
class GraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def stdlize_adj(adj: torch.Tensor):
    # 构造对称邻接(或权重)矩阵并加上自环
    out = adj.clone()
    out.fill_diagonal_(1.0)
    return torch.stack([out, out.T]).max(0).values


def norm_adj(adj):
    # 没有使用稀疏矩阵的更简单的版本，可能更慢
    D = adj.sum(1)**(-1)
    D = torch.diag(D)
    return torch.matmul(D, adj)
