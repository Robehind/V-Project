import os
import random
import torch
import models
import agents
import curriculums
import evalors
import environments as envs
from tqdm import tqdm
from tensorboardX import SummaryWriter
from environments.env_wrapper import make_envs, VecEnv
from utils.init_func import (
    get_args,
    make_exp_dir,
    set_seed,
    get_all_models
)
from vnenv.utils.record_utils import add_eval_data_seq
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
    env_args_list = env_cls.args_maker(args.env_args, args.proc_num)
    env_fns = [make_envs(e, env_cls) for e in env_args_list]
    Venv = VecEnv(env_fns)
    Venv.update_settings(args.eval_task)
    Venv.calc_shortest(args.calc_spl)

    # TODO init CLscher

    # 环境返回关于观察与动作的信息，方便初始化模型
    obs_info = Venv.shapes
    act_sz = Venv.action_sz

    # init model and load params
    model = model_cls(obs_info, act_sz, **args.model_args)
    if model is not None:
        model.eval()
        print(model)
    # init agent
    agent = agent_cls(model, Venv, args.gpu_ids, **args.agent_args)
    # make exp directory
    make_exp_dir(args, 'EvalAll-')

    tx_writer = SummaryWriter(log_dir=os.path.join(args.exp_dir, 'tblog'))
    paths_f = get_all_models(args)
    pbar = tqdm(total=len(paths_f), desc='Models')

    for p, model_id in paths_f:
        # TODO 重置环境的随机情况，可能不好？
        set_seed(args.seed)
        Venv.re_seed(args.seed)

        model.load_state_dict(torch.load(p))
        eval_data = eval_func(agent, Venv, args.eval_epi, bar_leave=False)
        # log
        add_eval_data_seq(tx_writer, eval_data, model_id)
        pbar.update(1)

    Venv.close()
    tx_writer.close()
    pbar.close()


if __name__ == "__main__":
    main()