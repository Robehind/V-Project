import torch
import os
import numpy as np


def norm_col_init(weights, std=1.0):
    """参数初始化，还没看是什么初始化"""
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def weights_init(m):
    """参数初始化，还没看是什么初始化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def save_model(model, path_to_save, title):
    """保存模型到path_to_save。保存的名称是title_{local_time}.dat,时间会自动算出来
       也可以用于保存优化算法的状态
    """
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    state_to_save = model.state_dict()
    import time
    start_time = time.time()
    time_str = time.strftime(
        "%H%M%S", time.localtime(start_time)
    )
    save_path = os.path.join(
        path_to_save,
        f"{title}_{time_str}.dat"
    )
    torch.save(state_to_save, save_path)


def optim2cuda(optim, gpu_id):
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(gpu_id)
