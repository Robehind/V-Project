import json


class VNENVargs:
    '''管理所有参数的类，默认参数在这里初始化'''
    def __init__(self, args_dict=None, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        self.update(args_dict, **kwargs)

    def update(self, args_dict=None, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        if args_dict is not None:
            for k in args_dict:
                setattr(self, k, args_dict[k])
        if kwargs is not None:
            for k in kwargs:
                setattr(self, k, kwargs[k])

    def save_args(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def env_args(self):
        return dict(
            dynamics_args=self.dynamics_args,
            obs_args=self.obs_args,
            event_args=self.event_args,
            seed=self.seed
        )


args_dict = dict(
    # general params
    verbose=False,  # 为True时将在控制台打印很多信息，调试用
    seed=1114,  # 随机数生成种子
    gpu_ids=-1,  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    load_model_dir='',  # 要读取的模型参数的完整路径，包括文件名
    load_optim_dir='',  # 要读取的优化其参数的完整路径，包括文件名
    exps_dir='../EXPS',  # 保存所有实验文件夹的路径
    exp_name='demo_exp',  # 将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    exp_dir='',  # 单次实验的完整路径，会根据时间自动生成
    proc_num=1,  # 进程数

    # train proc params
    total_train_steps=6e3,  # 指定训练多少frames
    print_freq=1000,  # 每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq=10000,  # 每进行n个episode，就保存一次模型参数

    # eval proc params
    total_eval_epi=1000,  # 指定测试时测试多少个episode
    best_a=False,  # 测试阶段是否取概率最高的动作，如果为false，那么就还是随机取
    results_json='result.json',  # 测试完成后结果输出到哪个文件

    # task params
    env='AbsEnv',
    event_args=dict(
        max_steps=50,  # 每个episode的最大步长，即agent在episode中的最大行动数
        reward_func={
            'collision': -0.1,
            'step': -0.01,
            'success': 10,
            'fail': 0
        }
    ),
    dynamics_args=dict(
        action_dict={
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'Done': None
        },
        scene_ids=[0, 1],
    ),
    obs_args=dict(
        obs_dict={
            'map': 'mat',
            'rela': 'relapos'
        },
    ),

    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=1,
        nsteps=1000,
        vf_param=0.5,
        ent_param=0,
    ),
    model='DemoModel',
    model_args={},
    agent='BaseAgent',
    agent_args={},
    optim='Adam',
    optim_args=dict(
        lr=0.0001,
    ),

    # exp params
    sampler='BaseSampler',
    sampler_args=dict(
        batch_size=20,
    ),
    CLscher='BaseCL',
    CLscher_args={},

    # train or eval funcs
    trainer='basic_train',
    evalor=''
)

args = VNENVargs(args_dict)
