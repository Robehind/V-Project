from vnenv.utils.default_args import args

args.update(
    # general params
    seed=1114,  # 随机数生成种子
    gpu_ids=[0],  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    load_model_dir='',  # 要读取的模型参数的完整路径，包括文件名
    load_optim_dir='',  # 要读取的优化其参数的完整路径，包括文件名
    exps_dir='../EXPS',  # 保存所有实验文件夹的路径
    exp_name='demo',  # 将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    exp_dir='',  # 单次实验的完整路径，会根据时间自动生成
    proc_num=16,  # 进程数

    # train proc params
    total_train_steps=1500000,  # 指定训练多少frames
    print_freq=1000,  # 每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq=100000,  # 每进行n个episode，就保存一次模型参数

    # eval proc params
    total_eval_epi=1000,  # 指定测试时测试多少个episode

    # task params
    event_args=dict(
        max_steps=50,  # 每个episode的最大步长，即agent在episode中的最大行动数
        reward_func={
            'collision': 0,
            'step': 0.02,
            'success': 10,
            'fail': 0
        }
    ),
    dynamics_args=dict(
        action_dict={
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'Done': None
        },
        scene_ids=[1],
    ),
    obs_args=dict(
        obs_dict={
            # 'rela': 'relapos',
            'map': 'mat'
        },
        map_sz=(10, 20)
    ),

    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=1,
        nsteps=1000,
        vf_param=0.5,
        ent_param=0,
    ),
    model='DemoModel',
    agent='BaseAgent',
    optim='Adam',
    optim_args=dict(
        lr=0.0007,
    ),

    # exp params
    sampler='BaseSampler',
    sampler_args=dict(
        batch_size=160,
    ),
)
