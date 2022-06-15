from .base_args import args

args.env_args.pop('target_embedding')
args.env_args.pop('wd_path')
args.update(
    exp_name='STD',
    # env params
    env_id='ZhuTdThor-v1',
    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01,
        optim='Adam',
        optim_args=dict(lr=0.0001,)),
    model='Zhu2017',
    model_args=dict(learnable_x=False),
)
