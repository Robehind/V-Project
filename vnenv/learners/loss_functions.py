import torch
import torch.nn.functional as F
from typing import Tuple


# define some general loss functions for learner to use
def _ac_loss(
    pi_batch: torch.Tensor,
    v_batch: torch.Tensor,
    td_target: torch.Tensor,
    a_batch: torch.Tensor,
    vf_param: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    adv = td_target - v_batch.detach()
    pi_a = F.softmax(pi_batch, dim=1).gather(1, a_batch)
    policy_loss = (-torch.log(pi_a) * adv).mean()
    # TODO 可以不用smooth l1
    value_loss = vf_param * F.smooth_l1_loss(v_batch, td_target)
    return policy_loss, value_loss
