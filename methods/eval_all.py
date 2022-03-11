import os
import random
import torch
import models
import agents
import taskers
import evalors
import json
import gym
import taskenvs
from gym.spaces import Dict as dict_spc
from tqdm import tqdm
from utils.init_func import (
    get_args,
    make_exp_dir,
    set_seed,
    get_all_models
)
os.environ["OMP_NUM_THREADS"] = '1'


def main():
    # 读取参数
    args = get_args(os.path.basename(__file__))
    # 随机数设定
    if args.seed == 1114:
        args.seed = random.randint(0, 9999)
    set_seed(args.seed)
    # 确认gpu可用情况
    if args.gpu_ids == -1:
        args.gpu_ids = None
    else:
        # TODO 在a2c中暂时只调用一块gpu用于测试，多线程训练可能需要调用pytorch本身的api
        assert torch.cuda.is_available()

    # 加载具体类
    model_cls = getattr(models, args.model)
    agent_cls = getattr(agents, args.agent)
    tasker_cls = getattr(taskers, args.tasker)
    eval_func = getattr(evalors, args.evalor)

    # gym 多进程化
    # 此时不对task space进行初始化
    Venv = gym.vector.make(
        args.env_id, num_envs=args.proc_num, **args.env_args)
    # 环境返回关于观察与动作的信息，方便初始化模型
    obs_spc = Venv.single_observation_space
    act_spc = Venv.single_action_space
    # 如果obs没有以dict形式组织，那么以关键字‘OBS’包装一下，且不改变obs_space
    if not isinstance(obs_spc, dict_spc):
        Venv = gym.wrappers.TransformObservation(Venv, lambda x: {'OBS': x})
    Venv.seed(args.seed)
    if args.extra_info is not None:
        Venv.call('add_extra_info', args.extra_info)

    # init tasker 此时通过调用Venv的call方法修改各个进程的task space
    _ = tasker_cls(Venv, args.eval_task, **args.tasker_args)

    # init model and load params
    model = model_cls(obs_spc, act_spc, **args.model_args)
    if model is not None:
        model.eval()
        print(model)
    # init agent
    agent = agent_cls(model, Venv, args.gpu_ids, **args.agent_args)
    # make exp directory
    make_exp_dir(args, 'EvalAll-')

    # tx_writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, 'tblog'))
    paths_f = get_all_models(args)
    pbar = tqdm(total=len(paths_f), desc='Models')

    all_trajs = []
    for p, model_id in paths_f:
        # TODO 重置环境的随机情况，可能不好？
        set_seed(args.seed)
        Venv.seed(args.seed)

        model.load_state_dict(torch.load(p))
        eval_trajs = eval_func(agent, Venv, args.eval_epi,
                               model_id=args.model+"_"+str(model_id),
                               bar_leave=False)
        all_trajs += eval_trajs
        pbar.update(1)
    with open(os.path.join(args.exp_dir, 'trajs.json'), "w") as fp:
        json.dump(all_trajs, fp)
    Venv.close()
    pbar.close()


if __name__ == "__main__":
    main()
