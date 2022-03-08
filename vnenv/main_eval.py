import os
import random
import torch
import gym
from gym.spaces import Dict as dict_spc
import json
import evalors
import models
import agents
import taskers
import taskenvs
from utils.init_func import (
    get_args,
    make_exp_dir,
    set_seed,
    load_or_find_model
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
    # TODO 测试和训练用的tasker可能不一样...
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
        print(model)
    load_dir = load_or_find_model(args)
    if load_dir != '':
        model.load_state_dict(torch.load(load_dir))
    model_id = os.path.split(load_dir)[-1].split('_')[:-1]
    model_id = '_'.join(model_id)
    model.eval()
    # init agent
    agent = agent_cls(model, Venv, args.gpu_ids, **args.agent_args)
    # make exp directory
    make_exp_dir(args, 'Eval-')
    # evaluating
    eval_trajs = eval_func(agent, Venv, args.eval_epi, model_id=model_id)
    Venv.close()
    with open(os.path.join(args.exp_dir, 'trajs.json'), "w") as fp:
        json.dump(eval_trajs, fp, indent=4)
    # output
    # if args.record_traj:
    #     trajs = eval_data.pop('trajs')
    #     with open(os.path.join(args.exp_dir, 'trajs.json'), "w") as fp:
    #         json.dump(trajs, fp, indent=4)
    # add_eval_data(writer, eval_data)


if __name__ == "__main__":
    main()
