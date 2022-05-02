from .base_args import args

args.update(
    exp_name='TgtAttActVecDmodel-lite-explr',
    # env params
    env_id='GradThor-v1',
    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01,
        optim='RMSprop',
        optim_args=dict(lr=0.0007,)),
    model='TAttAVecDmodel',
    model_args=dict(
        learnable_x=False, done_thres=0.5,
        done_net_path='./PriorDoneNet/DoneNet_4000.dat', init='zeros'),
    trainer='grad_train',
)
