from .base_args import args

args.update(
    exp_name='TgtAttActMat',
    # env params
    env_id='GradThor-v1',
    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01),
    model='TgtAttActMatModel',
    model_args=dict(learnable_x=False, init='zeros'),
    optim_args=dict(lr=0.0001)
)
