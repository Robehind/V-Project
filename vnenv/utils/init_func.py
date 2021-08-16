import sys
import importlib
import os
import time
import numpy as np
import torch
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # optional
    # torch.backends.cudnn.deterministic = TRue


def search_newest_model(exps_dir, exp_name):
    """根据所有实验的路径和实验名称搜索最新的模型"""
    _, dirs, _ = next(os.walk(exps_dir))
    tmp = -1
    dd = []
    for d in dirs:
        sp = d.split('_')
        if sp[0] == exp_name:
            dd.append(d)
    if dd == []:
        return None

    ff = None
    fd = None
    tmp = -1
    dd.sort()
    for d in dd[::-1]:
        _, _, files = next(os.walk(os.path.join(exps_dir, d)))
        for f in files:
            if f.split('.')[-1] == 'dat':
                frame = int(f.split('_')[1])

                if frame > tmp:
                    tmp = frame
                    ff = f
                    fd = d
        if ff is not None and fd is not None:
            return os.path.join(exps_dir, fd, ff)
    return None


def get_args(basename: str):
    try:
        conf_file = sys.argv[1]
    except IndexError:
        print(f'Usage: {basename} <configuration_file_name>')
        exit()
    module_str = conf_file.split(".")[0].replace('/', '.')
    args = importlib.import_module(module_str).args
    print(f'Loaded "{conf_file}", exp name: {args.exp_name}')
    return args


def make_exp_dir(args, prefix=''):
    """生成实验文件夹，会修改args.exp_dir, 并且会复制一次参数。可增加前缀"""
    start_time = time.time()
    time_str = time.strftime(
        "%y%m%d_%H%M%S", time.localtime(start_time)
    )
    args.exp_dir = os.path.join(
        args.exps_dir,
        prefix + args.exp_name + '_' + time_str
    )
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    # 保存本次实验的参数
    args.save_args(os.path.join(args.exp_dir, 'args.json'))


def load_or_find_model(args):
    """当load_model_dir不为空时，载入模型；否则自动寻找最新模型.结果储存在args中"""
    if args.load_model_dir != '':
        if os.path.exists(args.load_model_dir):
            print("load %s" % args.load_model_dir)
            # frames = int(os.path.basename(args.load_model_dir).split('_')[1])
        else:
            raise Exception(f'{args.load_model_dir} is not exists.')
    else:
        print('Didn\'t specify a trained model. Searching for a newest one')
        find_path = search_newest_model(args.exps_dir, args.exp_name)
        if find_path is not None:
            print("Searched the newest model: %s" % find_path)
            args.load_model_dir = find_path
        else:
            print("Can't find a newest model. Load Nothing.")
    return args.load_model_dir
