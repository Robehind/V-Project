from tqdm import tqdm
import numpy as np
from vnenv.utils.record_utils import MeanCalcer, data_output
from vnenv.environments import VecEnv


def basic_eval(
    args,
    agent,
    envs: VecEnv
):
    agent.model.eval()
    proc_num = args.proc_num
    total_epi = args.total_eval_epi

    epis = 0
    env_steps = np.zeros((proc_num))
    env_rewards = np.zeros((proc_num))
    false_action_ratio = [[] for _ in range(proc_num)]
    test_scalars = MeanCalcer()
    obs = envs.reset()

    pbar = tqdm(total=total_epi)
    while epis < total_epi:
        action = agent.action(obs)
        obs_new, r, done, info = envs.step(action)
        agent.reset_rct(done == 1)
        obs = obs_new
        env_rewards += r
        env_steps += 1
        for i in range(proc_num):
            t_info = info[i]
            if t_info is None:  # info is None means this proc does nothing
                continue
            false_action_ratio[i].append(
                t_info['false_action'] / env_steps[i]
            )
            if done[i] and epis < total_epi:
                epis += 1
                pbar.update(1)
                if args.calc_spl:
                    if t_info['success']:
                        assert t_info['min_len'] <= env_steps[i],\
                            f"{t_info['min_len']}>{env_steps[i]}"
                        # TODO spl计算问题。0？done？
                        spl = t_info['min_len']/env_steps[i]
                    else:
                        spl = 0
                data = {
                    'ep_length:': env_steps[i],
                    'SR:': t_info['success'],
                    'SPL:': spl,
                    'total_reward:': env_rewards[i],
                    'epis': 1,
                    'false_action_ratio': false_action_ratio[i]
                }
                test_scalars.add(data)
                env_steps[i] = 0
                env_rewards[i] = 0
                false_action_ratio[i] = []

    envs.close()
    pbar.close()
    data_output(args.exp_dir, args.results_json, test_scalars)
