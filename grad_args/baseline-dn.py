from .base_args import args

args.update(
    eval_epi=1000,
    exp_name='Baseline-DN',
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
    model='BaseDoneModel',
    model_args=dict(
        dropout_rate=0, learnable_x=False, done_thres=0.6,
        done_net_path='./PriorDoneNet/DoneNet_4000.dat', init='zeros'),
    optim_args=dict(lr=0.0001)
)
