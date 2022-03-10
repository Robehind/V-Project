from vnenv.utils.default_args import args

args.update(
    # general params
    seed=8952,  # 随机数生成种子
    gpu_ids=[0],  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    load_model_dir='',  # 要读取的模型参数的完整路径，包括文件名
    load_optim_dir='',  # 要读取的优化其参数的完整路径，包括文件名
    exps_dir='../demoEXPS',  # 保存所有实验文件夹的路径
    exp_name='thor-a2c-demo',  # 将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    exp_dir='',  # 单次实验的完整路径，会根据时间自动生成
    proc_num=8,  # 进程数
    extra_info=True,

    # train proc params
    train_steps=200000,  # 指定训练多少frames
    print_freq=1000,  # 每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq=50000,  # 每进行n个episode，就保存一次模型参数
    train_task={
        "scenes": {'kitchen': '25'},
        "targets": {'kitchen': ["Microwave", 'Sink']},
    },

    # eval proc params
    eval_epi=1000,  # 指定测试时测试多少个episode
    eval_task={
        "scenes": {'kitchen': '25'},
        "targets": {'kitchen': ["Microwave", 'Sink']},
    },
    evalor='basic_eval',

    # env params
    env_id='FcTdThor-v0',
    env_args={
        'actions': [
            'MoveAhead', 'RotateLeft',
            'RotateRight', 'Done'
        ],
        "reward_dict": {
            "collision": -0.01,
            "step": -0.01,
            "success": 1.0,
            "fail": -0.01,
        },
        'rotate_angle': 45,
        'max_steps': 100,
    },

    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0,
    ),
    model='FcLstmModel',
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
        batch_size=64,
        exp_length=8,
        buffer_limit=8
    ),
)
