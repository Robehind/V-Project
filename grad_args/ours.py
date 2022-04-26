from .base_args import args

args.env_args.update(
    actions=['MoveAhead', 'RotateLeft', 'RotateRight',
             'LookUp', 'LookDown'])
args.update(
    exps_dir='../grad_exps',
    exp_name='TgtAtt-baseGTD',
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
    model='TgtAttModel',
    model_args=dict(dropout_rate=0, learnable_x=False, init='zeros'),
    optim_args=dict(lr=0.0001)
)
