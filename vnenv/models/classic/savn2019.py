import torch
import torch.nn as nn
import torch.nn.functional as F
from vnenv.utils.net_utils import weights_init, norm_col_init


class SavnBase(torch.nn.Module):
    """"""
    def __init__(
        self,
        obs_shapes,
        act_sz,
        target_sz=300,
        dropout_rate=0.25,
    ):
        resnet_embedding_sz = 512
        hidden_state_sz = 512
        super(SavnBase, self).__init__()

        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.embed_glove = nn.Linear(target_sz, 64)
        self.embed_action = nn.Linear(act_sz, 10)

        pointwise_in_channels = 138

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.hidden_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
        self.rct_shapes = {'hx': (hidden_state_sz, ),
                           'cx': (hidden_state_sz, ),
                           'action_probs': (act_sz, )}
        dtype = next(self.lstm.parameters()).dtype
        self.rct_dtypes = {'hx': dtype, 'cx': dtype, 'action_probs': dtype}
        num_outputs = act_sz
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)
        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, act_sz)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
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
        self.dropout = nn.Dropout(p=dropout_rate)

    def embedding(self, state, target, action_probs):

        action_embedding_input = action_probs
        glove_embedding = F.relu(self.embed_glove(target))
        glove_reshaped = glove_embedding.view(-1, 64, 1, 1).repeat(1, 1, 7, 7)
        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = \
            action_embedding.view(-1, 10, 1, 1).repeat(1, 1, 7, 7)
        image_embedding = F.relu(self.conv1(state))
        x = self.dropout(image_embedding)
        x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
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
        target = obs['glove']
        action_probs = rct['action_probs']
        hx, cx = rct['hx'], rct['cx']

        x = self.embedding(state, target, action_probs)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))

        return dict(
            policy=actor_out,
            value=critic_out,
            rct=dict(
                hx=hx,
                cx=cx,
                action_probs=F.softmax(actor_out, dim=1).detach()
            )
        )