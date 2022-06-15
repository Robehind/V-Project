from .base_args import args

args.update(
    gpu_ids=[1],
    exp_name='Savnbase',
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
        optim='Adam',
        optim_args=dict(lr=0.0001,)),
    model='SavnBase',
    model_args=dict(dropout_rate=0.25, learnable_x=False),
)
