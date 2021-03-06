import json


class VNENVargs:
    '''管理所有参数的类，默认参数在这里初始化'''
    def __init__(self, args_dict=None, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        self.update(args_dict, True, **kwargs)

    def update(self, args_dict=None, init=False, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        if args_dict is not None:
            for k in args_dict:
                assert init or hasattr(self, k), f"don't have {k}"
                setattr(self, k, args_dict[k])
        if kwargs is not None:
            for k in kwargs:
                assert init or hasattr(self, k), f"don't have {k}"
                setattr(self, k, kwargs[k])

    def save_args(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)


args_dict = dict(
    # general params
    verbose=False,  # 为True时将在控制台打印很多信息，调试用
    seed=1114,  # 随机数生成种子
    gpu_ids=-1,  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    load_model_dir='',  # 要读取的模型参数的完整路径，包括文件名
    load_optim_dir='',  # 要读取的优化其参数的完整路径，包括文件名
    eval_all_dir='',  # 对某个文件夹下的所有模型进行评估的路径
    exps_dir='../EXPS',  # 保存所有实验文件夹的路径
    exp_name='demo_exp',  # 将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    exp_dir='',  # 单次实验的完整路径，会根据时间自动生成
    proc_num=1,  # 进程数
    debug=False,  # set detection anomaly True or False
    # add extra info by env if None, means no extra info, and don't call
    # add_extra_info

    # train proc params
    trainer='basic_train',
    train_extra_info=False,
    train_steps=6e3,  # 指定训练多少frames
    print_freq=1000,  # 每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq=10000,  # 每进行n个episode，就保存一次模型参数
    train_task=None,  # 训练环境设置

    # validate params 验证频率和模型保存频率是一样的
    validater='validate',  # 用哪个评估函数来验证
    val_extra_info=True,
    val_task=None,  # 验证环境设置
    val_epi=0,  # 验证的次数

    # eval proc params
    evalor='basic_eval',
    eval_extra_info=True,
    eval_epi=1000,  # 指定测试时测试多少个episode
    eval_task=None,  # 测试环境设置

    # env params
    env_id='TaskEnv',
    env_args={},

    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=1,
        nsteps=1000,
        vf_param=0.5,
        ent_param=0,
        optim='Adam',
        optim_args=dict(lr=0.0001,),
    ),
    model='DemoModel',
    model_args={},
    agent='BaseAgent',
    agent_args={},

    # exp params
    sampler='BaseSampler',
    sampler_args=dict(
        batch_size=20,
    ),
    recorder='BaseRecorder',
    tasker='Tasker',
    tasker_args={},
)

args = VNENVargs(args_dict)
