from methods.utils.default_args import args
targets = {
    'kitchen': [
        'Fridge', 'Microwave', 'Sink', 'GarbageCan', 'LightSwitch'],
    'living_room': [
        'Sofa', 'Television', 'Laptop', 'GarbageCan', 'LightSwitch'],
    'bedroom': [
        'Bed', 'AlarmClock', 'Laptop', 'GarbageCan', 'LightSwitch'],
    'bathroom': [
        'HandTower', 'SoapBottle', 'Sink', 'GarbageCan', 'LightSwitch']}

args.update(
    # general params
    seed=1014,
    gpu_ids=[0],
    proc_num=8,
    # train proc params
    train_steps=5e7,
    print_freq=20000,
    model_save_freq=1e6,
    train_task={
        "scenes": {'kitchen': '2,3,8,9,11,12,14,15,22-28',
                   'living_room': '2,5,8,10,11,13,19,21-26,28,29',
                   'bedroom': '1,2,4,7,10,12,13,15,16,18-20,22,26,29',
                   'bathroom': '2,5,9,11,13,15,17-20,22,23,25,26,29'},
        "targets": targets},
    # eval proc params
    eval_epi=2000,  # 指定测试时测试多少个episode
    eval_task={
        "scenes": {
            'kitchen': '5,6,17,19,29',
            'living_room': '4,6,7,16,20',
            'bedroom': '5,8,21,24,28',
            'bathroom': '8,12,14,21,28'},
        "targets": targets},
    # val proc params
    val_epi=0,
    val_task={
        "scenes": {
            'kitchen': '1,4,20,21,30',
            'living_room': '1,12,14,17,27',
            'bedroom': '3,6,14,17,27',
            'bathroom': '3,4,7,24,27'},
        "targets": targets},
    # env params
    env_args={
        'actions': [
            'MoveAhead', 'RotateLeft', 'RotateRight',
            'LookUp', 'LookDown', 'Done'],
        "reward_dict": {
            "collision": 0, "step": 0,
            "success": 5.0, "fail": 0},
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
