from tqdm import tqdm
from typing import Dict
import numpy as np
from methods.utils.record_utils import MeanCalcer
from gym.vector import VectorEnv


def validate(
    agent,
    envs: VectorEnv,
    total_epi: int
) -> Dict[float, list]:
    agent.model.eval()
    proc_num = envs.num_envs

    epis = 0
    env_steps = np.zeros((proc_num))
    env_rewards = np.zeros((proc_num))
    test_scalars = MeanCalcer()

    obs = envs.reset()
    done = np.ones((proc_num))

    pbar = tqdm(total=total_epi, desc='Validating', leave=False, unit='epi')
    while epis < total_epi:
        action, _ = agent.action(obs, done)
        obs_new, r, done, info = envs.step(action)
        obs = obs_new
        env_rewards += r
        env_steps += 1
        for i in range(proc_num):
            t_info = info[i]
            if t_info is None:  # info is None means this proc does nothing
                continue
            if done[i] and epis < total_epi:
                epis += 1
                pbar.update(1)
                data = {
                    'ep_length': env_steps[i],
                    'SR': t_info['success'],
                    'return': env_rewards[i],
                    'epis': 1
                }
                # 只要环境反馈了最短路信息，那么就算一下SPL
                if 'min_len' in t_info:
                    if t_info['success']:
                        assert t_info['min_acts'] <= env_steps[i],\
                            f"{t_info['min_acts']}>{env_steps[i]}"
                        # TODO spl计算问题。0？done？
                        spl = t_info['min_acts']/env_steps[i]
                    else:
                        spl = 0
                    data.update(dict(SPL=spl))
                test_scalars.add(data)
                env_steps[i] = 0
                env_rewards[i] = 0

    pbar.close()
    out = test_scalars.pop(['epis'])
    return out
