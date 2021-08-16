from ..default_args import args

args.update(
    seed=1,
    train_scenes={'kitchen': '25'},
    train_targets={'kitchen': ["Microwave", 'Sink']},
    test_scenes={'kitchen': '25'},
    test_targets={'kitchen': ["Microwave", 'Sink']},
    action_dict={
        'MoveAhead': ['m0'],
        'TurnLeft': ['r-45'],
        'TurnRight': ['r45'],
        'Done': None,
    },
    obs_dict={
        'fc': 'resnet50_fc_new.hdf5',
    },
    target_dict={
        'glove': '../thordata/word_embedding/word_embedding.hdf5',
    },
    grid_size=0.25,
    rotate_angle=45,
    total_train_frames=150000,
    total_eval_epi=1000,
    threads=16,
    exp_name='A2CLiteDemo',
    optimizer='Adam',
    model='FcLinearModel',
    agent='A2CAgent',
    runner='A2CRunner',
    loss_func='basic_loss',
    trainer='a2c_train',
    optim_args=dict(lr=0.0001,),
    print_freq=1000,
    max_epi_length=100,
    model_save_freq=150000,
    nsteps=80,
    gpu_ids=[0],
)
model_args_dict = dict(
    action_sz=len(args.action_dict),
    vobs_sz=2048,
    tobs_sz=300,
)
args.update(
    model_args=model_args_dict,
)
