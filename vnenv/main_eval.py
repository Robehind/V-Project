import os
import random
import torch
import evalors
import models
import agents
import curriculums
import environments as envs
from environments.env_wrapper import make_envs, VecEnv
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
    env_cls = getattr(envs, args.env)
    agent_cls = getattr(agents, args.agent)
    cl_cls = getattr(curriculums, args.CLscher)
    eval_func = getattr(evalors, args.evalor)

    # 生成多进程环境，每个进程环境初始化参数可能不一样
    # TODO 不同的进程加载不同的环境这种操作还是以后再弄吧
    env_args_list = env_cls.args_maker(args.env_args(False), args.proc_num)
    env_fns = [make_envs(e, env_cls) for e in env_args_list]

    # TODO 在使用随机测试时，环境自动随机初始状态，sche为空时需要继续操作
    # 当指定测试序列时，sche为空时进程需要等待退出
    no_op = False if args.CLscher == 'AbsCL' else True
    Venv = VecEnv(env_fns, min_len=args.calc_spl, no_sche_no_op=no_op)

    # init CLscher
    if args.CLscher != 'AbsCL':
        clscher = cl_cls(Venv, **args.CLscher_args)
        sche = clscher.sche
        if sche is not None and len(sche) != args.total_eval_epi:
            print("Warning: lengths of curriculums doesn't match the \
                  total eval epi number. Eval for the smaller number")
            args.total_eval_epi = min(args.total_eval_epi, len(sche))
            sche = sche[:args.total_eval_epi]
        Venv.sche_update(sche)

    # 环境返回关于观察与动作的信息，方便初始化模型
    obs_info = Venv.shapes
    act_sz = Venv.action_sz

    # init model and load params
    model = model_cls(obs_info, act_sz, **args.model_args)
    if model is not None:
        print(model)
    load_dir = load_or_find_model(args)
    if load_dir != '':
        model.load_state_dict(torch.load(load_dir))
    model.eval()
    # init agent
    agent = agent_cls(model, Venv, args.gpu_ids, **args.agent_args)
    # make exp directory
    make_exp_dir(args, 'TEST')
    # training
    eval_func(args, agent, Venv)


if __name__ == "__main__":
    main()
