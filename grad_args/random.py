from .base_args import args

args.update(
    exp_name='Random',
    proc_num=8,  # 进程数
    # env params
    env_id='FcTdThor-v1',
    agent='RandomAgent',
    # algo params
    model='BaseLstmModel',
    model_args=dict(dropout_rate=0, learnable_x=False, init='zeros'),
)
