from .base_args import args

args.update(
    exps_dir='../graduate',
    exp_name='a2c-yang',
    # env params
    env_id='FSTdThor-v1',
    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01),
    model='ScenePriors',
    model_args=dict(learnable_x=False),
    optim_args=dict(lr=0.0001)
)
