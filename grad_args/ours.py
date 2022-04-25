from .base_args import args

args.update(
    debug=True,
    sampler_args=dict(
        batch_size=20,
        exp_length=10,
        buffer_limit=2),
    proc_num=2,
    exps_dir='../grad_exps',
    exp_name='Amat_SplitD_TgtAtt',
    # env params
    env_id='FcTdThor-v1',
    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01),
    model='GradModel',
    model_args=dict(dropout_rate=0, learnable_x=True, init='randn'),
    optim_args=dict(lr=0.0001)
)
