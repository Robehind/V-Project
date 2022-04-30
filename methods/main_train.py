import os
import random
import torch
import gym
from gym.spaces import Dict as dict_spc
import trainers
import evalors
import models
import agents
import samplers
import learners
import taskenvs
from utils.init_func import get_args, make_exp_dir, set_seed
os.environ["OMP_NUM_THREADS"] = '1'


def main():
    # 读取、预处理参数
    args = get_args(os.path.basename(__file__))
    # 随机数设定
    if args.seed == 1114:
        args.seed = random.randint(0, 9999)
    set_seed(args.seed)
    # 确认gpu可用情况
    if args.gpu_ids == -1:
        args.gpu_ids = None
    else:
        # TODO 在a2c中暂时只调用一块gpu用于训练，多线程训练可能需要调用pytorch本身的api
        assert torch.cuda.is_available()

    # 加载具体类
    model_cls = getattr(models, args.model)
    agent_cls = getattr(agents, args.agent)
    sampler_cls = getattr(samplers, args.sampler)
    recorder_cls = getattr(samplers, args.recorder)
    learner_cls = getattr(learners, args.learner)
    tasker_cls = getattr(taskenvs, args.tasker)
    train_func = getattr(trainers, args.trainer)
    val_func = getattr(evalors, args.validater)

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

    # init recorder
    recorder = recorder_cls(Venv, args.train_extra_info)

    # TODO params management
    # init tasker 此时通过调用Venv的call方法修改各个进程的task space
    tasker = tasker_cls(Venv, args.train_task, **args.tasker_args)

    # init model and load params
    model = model_cls(obs_spc, act_spc, **args.model_args)
    if model is not None:
        print(model)
    # TODO 读取存档点，读取最新存档模型的参数到model
    if args.load_model_dir != '':
        print("load %s" % args.load_model_dir)
        model.load_state_dict(torch.load(args.load_model_dir))
    model.train()

    # init agent
    agent = agent_cls(model, Venv, args.gpu_ids, **args.agent_args)

    # init sampler
    sampler = sampler_cls(Venv, agent, recorder, **args.sampler_args)
    # init learner
    learner = learner_cls(model, **args.learner_args)
    # make exp directory
    if not args.debug:
        make_exp_dir(args)
    # training
    print('Set detect anomaly:', args.debug)
    with torch.autograd.set_detect_anomaly(args.debug):
        train_func(args, sampler, learner, tasker, val_func)


if __name__ == "__main__":
    main()
