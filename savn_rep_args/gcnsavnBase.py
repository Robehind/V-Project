from .base_args import args

wd_path = args.env_args['wd_path']
args.update(
    gpu_ids=[0],
    exp_name='GcnSavnBase',
    # env params
    env_id='FmScoreTdThor-v0',
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
    model='GcnSavnBase',
    model_args=dict(
        gcn_path='/mnt/ssd/vdata/gcn',
        wd_path=wd_path,
        dropout_rate=0.25, learnable_x=False),
)
