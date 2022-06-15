import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.net_utils import norm_col_init
from gym.spaces import Dict as Dictspc
from gym.spaces import Discrete
import numpy as np
from torch.nn.parameter import Parameter
from ..select_funcs import policy_select
from ..mem_infer import YangGCN


class GcnSavnBase(torch.nn.Module):
    """"""
    def __init__(
        self,
        obs_spc: Dictspc,
        act_spc: Discrete,
        gcn_path,
        wd_path,
        dropout_rate,
        learnable_x,
        gcn_embed_sz=512
    ):
        resnet_embedding_sz = 512
        hidden_state_sz = 512
        act_sz = act_spc.n
        target_sz = np.prod(obs_spc['wd'].shape)
        score_sz = np.prod(obs_spc['score'].shape)
        super(GcnSavnBase, self).__init__()

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.embed_glove = nn.Linear(target_sz, 64)
        self.embed_action = nn.Linear(act_sz, 10)
        pointwise_in_channels = 138 + 64

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)
        self.yangcn = YangGCN(
            gcn_path=gcn_path, wd_path=wd_path,
            output_sz=gcn_embed_sz, input_sz=score_sz, wd_type="fasttext")
        self.gcn_embed = nn.Linear(gcn_embed_sz, 64)

        lstm_input_sz = 7 * 7 * 64

        self.hidden_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        self.rct_shapes = {'hx': (hidden_state_sz, ),
                           'cx': (hidden_state_sz, ),
                           'action_probs': (act_sz, )}
        self.hx = Parameter(torch.zeros(1, self.hidden_sz), learnable_x)
        self.cx = Parameter(torch.zeros(1, self.hidden_sz), learnable_x)
        self.action_probs = Parameter(torch.zeros(1, act_sz), False)
        dtype = next(self.lstm.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype, 'action_probs': dtype}
        num_outputs = act_sz
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        # self.apply(weights_init)
        # relu_gain = nn.init.calculate_gain("relu")
        # self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
             self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)
        self.dropout = nn.Dropout(p=dropout_rate)

    def embedding(self, state, score, target, action_probs):

        action_embedding_input = action_probs
        glove_embedding = F.relu(self.embed_glove(target))
        glove_reshaped = glove_embedding.view(-1, 64, 1, 1).repeat(1, 1, 7, 7)
        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = \
            action_embedding.view(-1, 10, 1, 1).repeat(1, 1, 7, 7)
        image_embedding = F.relu(self.conv1(state))

        gcn_feat = self.yangcn(score)
        gcn_feat = F.relu(self.gcn_embed(gcn_feat), True)
        gcn_reshaped = gcn_feat.view(-1, 64, 1, 1).repeat(1, 1, 7, 7)
        x = self.dropout(image_embedding)
        x = torch.cat(
            (x, gcn_reshaped, glove_reshaped, action_reshaped), dim=1)
        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)
        return out

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)

        return actor_out, critic_out, (hx, cx)

    def forward(self, obs, rct):

        state = obs['res18fm']
        score = obs['score']
        target = obs['wd']
        action_probs = rct['action_probs']
        hx, cx = rct['hx'], rct['cx']

        x = self.embedding(state, score, target, action_probs)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))

        out = dict(
            policy=actor_out,
            value=critic_out,
            rct=dict(
                hx=hx, cx=cx,
                action_probs=F.softmax(actor_out, dim=1)  # .detach()
            )
        )
        out['action'] = policy_select(out).detach()
        return out
