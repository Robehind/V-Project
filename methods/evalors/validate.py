from collections import defaultdict
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
    pmarker = [defaultdict(int) for _ in range(proc_num)]
    explore = np.zeros((proc_num))
    collision = np.zeros((proc_num))
    test_scalars = MeanCalcer()

    obs = envs.reset()
    vis_cnt = []
    for i, info in enumerate(envs.call("info")):
        vis_cnt.append(info['visible'])
        pmarker[i][info['pose']] = 1
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
            if not pmarker[i][t_info['pose']]:
                explore[i] += 1
                pmarker[i][t_info['pose']] = 1
            if t_info['event'] == 'collision':
                collision[i] += 1
            vis_cnt[i] += int(t_info['visible'])
            if t_info is None:  # info is None means this proc does nothing
                continue
            if done[i] and epis < total_epi:
                vis_cnt[i] -= int(t_info['visible'])
                epis += 1
                pbar.update(1)
                data = {
                    'ep_length': env_steps[i],
                    'SR': t_info['success'],
                    'return': env_rewards[i],
                    'vis_cnt': vis_cnt[i],
                    'CR': collision[i] / env_steps[i],
                    'ER': explore[i] / env_steps[i]}
                # 只要环境反馈了最短路信息，那么就算一下SPL
                if 'min_acts' in t_info:
                    spl = 0
                    if t_info['success']:
                        assert t_info['min_acts'] <= env_steps[i],\
                            f"{t_info['min_acts']}>{env_steps[i]}"
                        # TODO spl计算问题。0？done？
                        spl = t_info['min_acts']/env_steps[i]
                    data.update(dict(SPL=spl))
                test_scalars.add(data)
                env_steps[i], env_rewards[i] = 0, 0
                collision[i], explore[i] = 0, 0
                pmarker[i] = defaultdict(int)
                # 需要获得reset后的info
                res_info = envs.call("info")[i]
                vis_cnt[i] = int(res_info['visible'])
                pmarker[i][res_info['pose']] = 1
    pbar.close()
    out = test_scalars.pop()
    return out
