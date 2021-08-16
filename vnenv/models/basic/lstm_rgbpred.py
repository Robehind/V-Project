import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_model import SimpleMP
import vnenv.models.perception as perception
from vnenv.utils.net_utils import weights_init


class LstmRGBpred(nn.Module):
    """RGB预测作为在线辅助任务的Lstm"""
    def __init__(
        self,
        action_sz,
        threads = 16,
        rgbloss_p = 0.0001,
        enc = 'SplitNetCNN',
        dec = 'SplitRGBPred'
        # state_sz,
        # target_sz,
    ):
        super(LstmRGBpred, self).__init__()
        # perception
        tobs_sz = 300
        self.threads = threads
        self.rgbloss_p = rgbloss_p
        enc = getattr(perception, enc)()
        dec = getattr(perception, dec)()

        self.conv_out_sz = enc.out_fc_sz(128, 128)
        self.vobs_conv = enc
        self.RGBpred = dec
        self.apply(weights_init)

        self.MP = SimpleMP(action_sz, self.conv_out_sz, tobs_sz, mode = 1)
        self.hidden_sz = self.MP.hidden_sz

        # normalize
        # mean = torch.FloatTensor([0.5269, 0.4565, 0.3687]).view(1,3,1,1)
        # self.mean = torch.nn.Parameter(mean)
        # self.mean.requires_grad = False
        # std = torch.FloatTensor([0.0540, 0.0554, 0.0567]).view(1,3,1,1)
        # self.std = torch.nn.Parameter(std)
        # self.std.requires_grad = False
    def recons(self,model_input):

        vobs = model_input['image'].permute(0,3,1,2)/ 255.
        vobs_embed = self.vobs_conv(vobs)

        RGB = self.RGBpred(vobs_embed).permute(0,2,3,1).squeeze().detach().numpy()

        return RGB[:,:,::-1]

    def forward(self, model_input):
        vobs = model_input['image'].permute(0,3,1,2)/ 255.

        # n_vobs = (vobs-self.mean)/self.std

        vobs_embed = self.vobs_conv(vobs)
        vobs_embed_f = torch.flatten(vobs_embed, 1)

        RGBloss = 0
        # 有batch的时候说明需要计算辅助loss了，采样的时候就不计算，节约一点
        if vobs_embed.shape[0] > self.threads:
            RGBloss = self.rgbloss_p*F.l1_loss(self.RGBpred(vobs_embed),vobs.detach(),reduction='sum')

        out = self.MP(vobs_embed_f, model_input['glove'], model_input['hidden'])
        out.update(rgb_loss=RGBloss)
        return out
