from .base_args import args

args.update(
    exp_name='TgtAttDmodel-with-hc',
    # env params
    env_id='FcTdThor-v1',
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
    model='TgtAttDmodel',
    model_args=dict(
        learnable_x=False, done_thres=0.5,
        done_net_path='./PriorDoneNet/DoneNet_4000.dat', init='zeros'),
)
