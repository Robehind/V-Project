from .base_args import args

args.update(
    exps_dir='../grad_exps',
    exp_name='a2c-basecnn',
    proc_num=8,  # 进程数
    # env params
    env_id='FrameTdThor-v0',
    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01),
    model='CNNmodel',
    model_args=dict(dropout_rate=0, learnable_x=False, init='zeros'),
    optim_args=dict(lr=0.0003)
)
