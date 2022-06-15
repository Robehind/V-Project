import torch
import torch.nn as nn
import torch.nn.functional as F
from methods.utils.net_utils import norm_col_init, weights_init
import scipy.sparse as sp
import numpy as np
import os
import h5py
from torch.nn.parameter import Parameter
from gym.spaces import Dict as Dictspc
from gym.spaces import Discrete
from ..select_funcs import policy_select
from torch.nn.functional import one_hot
from ..mem_infer.gcn import GraphConv


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class MJOBASE(torch.nn.Module):
    def __init__(
        self,
        obs_spc: Dictspc,
        act_spc: Discrete,
        dropout_rate,
        learnable_x,
        gcn_path="../vdata/gcn/",
        wd_type="glove",
        wd_path='../vdata/word_embedding/word_embedding.hdf5',
    ):
        action_space = act_spc.n
        hidden_state_sz = 512
        super(MJOBASE, self).__init__()

        # get and normalize adjacency matrix.
        # np.seterr(divide='ignore')
        adj_path = os.path.join(gcn_path, 'adjmat.dat')
        obj_path = os.path.join(gcn_path, 'objects.txt')
        A_raw = torch.load(adj_path)
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))

        n = int(A.shape[0])
        self.n = n

        self.embed_action = nn.Linear(action_space, 10)

        lstm_input_sz = 10 + n * 5 + 512

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        self.rct_shapes = {'hx': (hidden_state_sz, ),
                           'cx': (hidden_state_sz, ),
                           'action_probs': (action_space, )}
        self.hx = Parameter(torch.zeros(1, self.hidden_state_sz), learnable_x)
        self.cx = Parameter(torch.zeros(1, self.hidden_state_sz), learnable_x)
        self.action_probs = Parameter(torch.zeros(1, action_space), False)
        dtype = next(self.lstm.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype, 'action_probs': dtype}
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=dropout_rate)

        # glove embeddings for all the objs.
        with open(obj_path) as f:
            objects = f.readlines()
            self.objects = [o.strip() for o in objects]
        all_wds = torch.zeros(n, 300)
        wdf = h5py.File(wd_path, "r")
        wd = wdf[wd_type]
        for i in range(n):
            all_wds[i, :] = torch.from_numpy(wd[self.objects[i]][:])
        wdf.close()

        self.cos = torch.nn.CosineSimilarity(dim=-1)

        self.all_glove = nn.Parameter(all_wds)
        self.all_glove.requires_grad = False

        sz = self.n + 300
        self.W0 = GraphConv(sz, sz, False)  # nn.Linear(sz, sz, bias=False)
        self.W1 = GraphConv(sz, sz, False)  # nn.Linear(sz, sz, bias=False)
        self.W2 = GraphConv(sz, 5, False)  # nn.Linear(sz, 5, bias=False)
        self.W3 = GraphConv(10, 1, False)  # nn.Linear(10, 1, bias=False)

        self.final_mapping = nn.Linear(n, 512)

    def new_gcn_embed(self, objstate, class_onehot):
        batch_sz = class_onehot.shape[0]
        class_word_embed = torch.cat(
            (class_onehot.repeat(self.n, 1, 1).permute(1, 0, 2),
             self.all_glove.repeat(batch_sz, 1, 1)), dim=2)
        # class_word_embed = torch.cat(
        #     (class_onehot.repeat(self.n, 1), self.all_glove.detach()), dim=1)
        x = F.relu(self.W0(class_word_embed, self.A))
        x = F.relu(self.W1(x, self.A))
        x = F.relu(self.W2(x, self.A))
        x = torch.cat((x, objstate), dim=2)
        x = F.relu(self.W3(x, self.A)).squeeze(-1)
        # x = x.view(1, self.n)
        x = self.final_mapping(x)
        return x

    def embedding(self, target, action_probs, objstate):
        target_wd = self.all_glove[target.view(-1)]
        glove_sim = self.cos(
            self.all_glove.unsqueeze(0), target_wd.unsqueeze(1))
        class_onehot = one_hot(target.view(-1), num_classes=self.n)
        objstate = torch.cat((objstate, glove_sim.unsqueeze(-1)), dim=2)
        # objstate, class_onehot = self.list_from_raw_obj(objbb, target)
        action_embedding_input = action_probs
        action_embedding = F.relu(self.embed_action(action_embedding_input))
        x = objstate
        x = x.flatten(start_dim=1)  # x.view(1, -1)
        x = torch.cat((x, action_embedding), dim=1)
        out = torch.cat((x, self.new_gcn_embed(objstate, class_onehot)), dim=1)

        return out, None

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)

    def forward(self, obs, rct):
        objbb = obs['bbox']
        (hx, cx) = rct['hx'], rct['cx']

        target = obs['idx']
        action_probs = rct['action_probs']
        x, _ = self.embedding(target, action_probs, objbb)
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
