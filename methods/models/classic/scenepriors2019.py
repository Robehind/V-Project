from ..mem_infer import YangGCN
from ..plan import AClinear
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class ScenePriors(nn.Module):
    def __init__(
        self,
        obs_spc,
        act_spc,
        infer_sz=512,
        vobs_embed_sz=512,
        tobs_embed_sz=512,
        gcn_embed_sz=512
    ) -> None:
        super().__init__()
        vobs_sz = np.prod(obs_spc['fc'].shape)
        tobs_sz = np.prod(obs_spc['glove'].shape)
        score_sz = np.prod(obs_spc['score'].shape)
        self.v_embed = nn.Linear(vobs_sz, vobs_embed_sz)
        self.t_embed = nn.Linear(tobs_sz, tobs_embed_sz)

        self.yanggcn = YangGCN(output_sz=gcn_embed_sz, input_sz=score_sz)
        self.lstm = nn.LSTMCell(
            vobs_embed_sz+tobs_embed_sz+gcn_embed_sz,
            infer_sz
        )
        self.rct_shapes = {'hx': (infer_sz, ), 'cx': (infer_sz, )}
        dtype = next(self.lstm.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype}
        self.plan = AClinear(infer_sz, act_spc.n)

    def forward(self, obs, rct):
        x_fc = obs['fc']
        x_score = obs['score']
        x_tgt = obs['glove']
        hx, cx = rct['hx'], rct['cx']
        x_fc = F.relu(self.v_embed(x_fc))
        x_tgt = F.relu(self.t_embed(x_tgt))
        x_gcn = F.relu(self.yanggcn(x_score))
        embed = torch.cat((x_fc, x_tgt, x_gcn), dim=1)
        hx, cx = self.lstm(embed, (hx, cx))
        out = self.plan(hx)
        out['rct'] = dict(hx=hx, cx=cx)
        return out
