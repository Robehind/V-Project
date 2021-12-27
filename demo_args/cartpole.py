from vnenv.utils.default_args import args

args.update(
    # general params
    seed=1114,  # 随机数生成种子
    gpu_ids=-1,  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    load_model_dir='',  # 要读取的模型参数的完整路径，包括文件名
    load_optim_dir='',  # 要读取的优化其参数的完整路径，包括文件名
    exps_dir='../demoEXPS',  # 保存所有实验文件夹的路径
    exp_name='cartpole',  # 将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    exp_dir='',  # 单次实验的完整路径，会根据时间自动生成
    proc_num=4,  # 进程数

    # train proc params
    train_steps=300000,  # 指定训练多少frames
    print_freq=2000,  # 每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq=300000,  # 每进行n个episode，就保存一次模型参数
    train_task={},

    # validate params
    val_mode=False,
    val_task={},
    val_epi=1000,
    # validater='gym_eval',

    # eval proc params
    eval_epi=1000,  # 指定测试时测试多少个episode
    eval_task={},
    record_traj=True,
    # evalor='gym_eval',

    # task params
    env='CartPolev1',
    dynamics_args={},
    obs_args={'key': 'obs'},
    event_args={'r_scale': 100.0},

    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.98,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=1,
        ent_param=0,
    ),
    model='CartModel',
    agent='BaseAgent',
    agent_args=dict(
        select_func='policy_select'
    ),
    optim='Adam',
    optim_args=dict(
        lr=0.0001,
    ),

    # exp params
    sampler='BaseSampler',
    sampler_args=dict(
        batch_size=20,
        exp_length=5,
        buffer_limit=4
    ),
)