from tqdm import tqdm
from typing import Dict
import numpy as np
from gym.vector import VectorEnv


def basic_eval(
    agent,
    envs: VectorEnv,
    total_epi: int,
    model_id: str,
    bar_leave: bool = True,
    bar_desc: str = '',
) -> Dict[float, list]:
    agent.model.eval()
    proc_num = envs.num_envs

    epis = 0
    env_rewards = np.zeros((proc_num))

    acts_rec = [[] for _ in range(proc_num)]
    poses_rec = [[] for _ in range(proc_num)]
    events_rec = [[] for _ in range(proc_num)]
    trajs = []

    obs = envs.reset()
    done = np.ones((proc_num))

    pbar = tqdm(total=total_epi, desc=bar_desc, leave=bar_leave, unit='epi')
    while epis < total_epi:
        action, _ = agent.action(obs, done)
        obs_new, r, done, info = envs.step(action)
        obs = obs_new
        env_rewards += r
        for i in range(proc_num):
            t_info = info[i]
            acts_rec[i].append(int(action[i]))
            poses_rec[i].append(t_info['pose'])
            events_rec[i].append(t_info['event'])
            if t_info is None:  # info is None means this proc does nothing
                continue
            if done[i] and epis < total_epi:
                epis += 1
                pbar.update(1)
                poses_rec[i] = [t_info['start_at']] + poses_rec[i]
                traj = {
                    'scene': t_info['scene'],
                    'target': t_info['target'],
                    'model': model_id,
                    'success': int(t_info['success']),
                    'return': env_rewards[i],
                    'actions': acts_rec[i].copy(),  # A_0 to A_T-1, total of T
                    'poses': poses_rec[i].copy(),  # S_0 to S_T, total of T + 1
                    'events': events_rec[i].copy(),  # E_1 to E_T, total of T
                    'agent_done': t_info['agent_done'],
                    'ep_length': len(acts_rec[i])
                }
                if 'min_acts' in t_info:
                    traj['min_acts'] = t_info['min_acts']
                trajs.append(traj)
                acts_rec[i], poses_rec[i], events_rec[i] = [], [], []
                env_rewards[i] = 0

    pbar.close()
    return trajs
