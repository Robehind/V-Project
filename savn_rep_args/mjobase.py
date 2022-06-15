from .base_args import args

args.env_args.pop('target_embedding')
args.env_args.pop('wd_path')
args.update(
    # general params
    exp_name='MJOBASE',
    # env params
    env_id='MjoThor-v0',
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
    model='MJOBASE',
    model_args=dict(
        gcn_path="../vdata/gcn/",
        wd_type="fasttext",
        wd_path='../vdata/word_embedding/word_embedding.hdf5',
        dropout_rate=0,
        learnable_x=False),
)
