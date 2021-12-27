import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from vnenv.utils.record_utils import MeanCalcer
from vnenv.models.basic.model4gym import CartModel
from vnenv.learners import A2CLearner
from vnenv.agents import BaseAgent
from vnenv.samplers import BaseSampler
from vnenv.environments.env_wrapper import VecEnv, make_envs
from vnenv.environments.gym_envs import CartPolev1
from collections import deque
import gym
import numpy as np
import time


# Hyperparameters
proc_num = 8
exp_length = 4
learning_rate = 0.001
gamma = 0.98
max_train_steps = 100000
PRINT_INTERVAL = 2000
reward_scale = 100.0
speedrun = True


def visualize(agt):
    v_env = gym.make('CartPole-v1')
    for _ in range(5):
        s = v_env.reset()
        d = False
        rr = 0
        while not d:
            v_env.render()
            a, _ = agt.action({
                'obs': np.array(s).reshape(1, -1)}, np.ones((1)))
            s, r, d, info = v_env.step(a[0])
            rr += 1
        print("Returns:", rr)
    v_env.close()


if __name__ == '__main__':
    env_args = CartPolev1.args_maker(
        {'event_args': {'r_scale': reward_scale}}, proc_num)
    env_fns = [make_envs(args, CartPolev1) for args in env_args]
    envs = VecEnv(env_fns)
    batch_sz = exp_length*proc_num

    model = CartModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    learner = A2CLearner(model, optimizer, gamma, gae_lbd=1,
                         vf_nsteps=float("inf"), vf_param=1,
                         vf_loss='smooth_l1_loss',
                         grad_norm_max=float("inf"),
                         batch_loss_mean=True)
    agt = BaseAgent(model, envs, None, select_func='policy_select')
    spler = BaseSampler(envs, agt, batch_size=exp_length*proc_num,
                        exp_length=exp_length, buffer_limit=proc_num)

    pbar = tqdm(total=max_train_steps)
    if not speedrun:
        writer = SummaryWriter("../cartpole")
    tracker = MeanCalcer()
    step_idx = 0

    rs = deque(maxlen=100)
    if speedrun:
        st = time.time()
    while step_idx < max_train_steps:
        step_idx += batch_sz
        exps = spler.sample()
        loss = learner.learn(exps)
        tracker.add(loss)
        if speedrun:
            data = spler.pop_records()
            for _ in range(int(data['epis'])):
                rs.append(data['return'])
            if sum(rs) >= 400:
                break
        pbar.update(batch_sz)
        if not speedrun and step_idx % PRINT_INTERVAL == 0:
            out = tracker.pop()
            out.update(spler.pop_records())
            for k, v in out.items():
                writer.add_scalar(k, v, step_idx)

    if not speedrun:
        writer.close()
    pbar.close()
    if speedrun:
        print(f'Finished. Time:{time.time() - st}')
    envs.close()
    visualize(agt)
