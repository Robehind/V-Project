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


# Hyperparameters
n_train_processes = 4
learning_rate = 0.0001
update_interval = 5
gamma = 0.98
max_train_steps = 300000
PRINT_INTERVAL = 2000
reward_scale = 100.0

if __name__ == '__main__':
    env_args = CartPolev1.args_maker(
        {'event_args': {'r_scale': reward_scale}}, n_train_processes)
    env_fns = [make_envs(args, CartPolev1) for args in env_args]
    envs = VecEnv(env_fns)

    model = CartModel(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    learner = A2CLearner(model, optimizer, gamma, 1, float("inf"), 1)
    agt = BaseAgent(model, envs, None, select_func='policy_select')
    spler = BaseSampler(envs, agt, 20, 5, 4)

    pbar = tqdm(total=max_train_steps)
    writer = SummaryWriter("../cartpole")
    tracker = MeanCalcer()
    step_idx = 0

    rs = deque(maxlen=100)
    while step_idx < max_train_steps:
        step_idx += update_interval*n_train_processes
        exps = spler.sample()
        loss = learner.learn(exps)
        tracker.add(loss)

        pbar.update(update_interval*n_train_processes)
        if step_idx % PRINT_INTERVAL == 0:
            out = tracker.pop()
            out.update(spler.pop_records())
            for k, v in out.items():
                writer.add_scalar(k, v, step_idx)
    pbar.close()
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
        print("reward", rr)
    envs.close()
