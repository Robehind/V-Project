from methods.utils.default_args import args

args.update(
    # general params
    seed=1114,  # 随机数生成种子
    gpu_ids=[0],  # 指定要使用的显卡，为-1时使用cpu。gpu_ids = [0,1,2,3]
    exps_dir='../grad_exps/base_exps',  # 保存所有实验文件夹的路径
    exp_name='a2c-base-dropout',  # 将用于生成本次实验的实验文件夹的文件名，因此尽量不要包含特殊符号
    proc_num=8,  # 进程数

    # train proc params
    train_steps=1e8,  # 指定训练多少frames
    print_freq=20000,  # 每进行n个frames，就向tensorboardx输出一次训练信息
    model_save_freq=1e7,  # 每进行n个episode，就保存一次模型参数
    train_task={
        "scenes": {'kitchen': '1-20', 'living_room': '1-20',
                   'bedroom': '1-20', 'bathroom': '1-20'},
        "targets": {
            'kitchen': [
                "Toaster", "Microwave", "Fridge",
                "CoffeeMaker", "GarbageCan", "Box", "Bowl"],
            'living_room': [
                "Pillow", "Laptop", "Television",
                "GarbageCan", "Box", "Bowl"],
            'bedroom': ["HousePlant", "Lamp", "Book", "AlarmClock"],
            'bathroom': ["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"],
        },
    },

    # eval proc params
    eval_epi=2000,  # 指定测试时测试多少个episode
    eval_task={
        "scenes": {'kitchen': '26-30', 'living_room': '26-30',
                   'bedroom': '26-30', 'bathroom': '26-30'},
        "targets": {
            'kitchen': [
                "Toaster", "Microwave", "Fridge",
                "CoffeeMaker", "GarbageCan", "Box", "Bowl"],
            'living_room': [
                "Pillow", "Laptop", "Television",
                "GarbageCan", "Box", "Bowl"],
            'bedroom': ["HousePlant", "Lamp", "Book", "AlarmClock"],
            'bathroom': ["Sink", "ToiletPaper", "SoapBottle", "LightSwitch"]}},
    evalor='basic_eval',

    # env params
    env_id='FcTdThor-v1',
    env_args={
        'actions': [
            'MoveAhead', 'RotateLeft', 'RotateRight',
            'LookUp', 'LookDown', 'Done'
        ],
        "reward_dict": {
            "collision": -0.1,
            "step": -0.01,
            "success": 5.0,
            "fail": -0.01,
        },
        'rotate_angle': 45,
        'max_steps': 200},

    # algo params
    learner='A2CLearner',
    learner_args=dict(
        gamma=0.99,
        gae_lbd=1,
        vf_nsteps=float("inf"),
        vf_param=0.5,
        ent_param=0.01,),
    model='BaseLstmModel',
    model_args=dict(dropout_rate=0.25),
    agent='BaseAgent',
    optim='Adam',
    optim_args=dict(lr=0.0001,),
    # exp params
    sampler='BaseSampler',
    sampler_args=dict(
        batch_size=320,
        exp_length=40,
        buffer_limit=8),
)
