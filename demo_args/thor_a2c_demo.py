from methods.utils.default_args import args

args.update(
    # general params
    seed=1114,  # 随机数生成种子
    gpu_ids=[0],  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    exps_dir='../demoEXPS',  # 保存所有实验文件夹的路径
    exp_name='thor-a2c-demo',  # 将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    proc_num=8,  # 进程数

    # train proc params
    train_steps=200000,  # 指定训练多少frames
    print_freq=1000,  # 每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq=50000,  # 每进行n个episode，就保存一次模型参数
    train_task={
        "scenes": {'kitchen': '25'},
        "targets": {'kitchen': ["Microwave", 'Sink']},
    },

    val_epi=0,
    val_task={
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
        'ctl_data_dir': '../vdata/thordata',
        'wd_path': '../vdata/word_embedding/word_embedding.hdf5',
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
        optim='Adam',
        optim_args=dict(lr=0.0001,),
    ),
    model='FcLstmModel',
    agent='BaseAgent',

    # exp params
    sampler='BaseSampler',
    sampler_args=dict(
        batch_size=64,
        exp_length=8,
        buffer_limit=8
    ),
)
