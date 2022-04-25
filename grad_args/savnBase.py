from .base_args import args

args.update(
    gpu_ids=[1],  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    exps_dir='../grad_exps',  # 保存所有实验文件夹的路径
    exp_name='savnbase',
    # env params
    env_id='FmTdThor-v0',
    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01,
    ),
    model='SavnBase',
    model_args=dict(dropout_rate=0, learnable_x=False),
    optim_args=dict(lr=0.0001),
)
