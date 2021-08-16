from vnenv.environments import DiscreteEnvironment
import cv2
params = {
    "offline_data_dir": '../thordata/mixed_offline_data',
    'action_dict': {
        'MoveAhead': ['m0'],
        'TurnLeft': ['r-45'],
        'TurnRight': ['r45'],
        'LookUp': ['p-30'],
        'LookDown': ['p30'],
        'Done': None
    },
    "target_dict": {
        "glove": '../thordata/word_embedding/word_embedding.hdf5',
    },
    "obs_dict": {
        "RGB": "images.hdf5",
        # 'res18fm':'resnet18_featuremap.hdf5'
    },
    "reward_dict": {
        "collision": -0.1,
        "step": -0.01,
        "SuccessDone": 10.0,
        "FalseDone": 0,
    },
    'max_steps': 1000,
    'grid_size': 0.25,
    'rotate_angle': 45,
    'move_angle': 45,
    'horizon_angle': 30,
    "chosen_scenes": ['FloorPlan222_physics'],
    "chosen_targets": None,
    "debug": False,
}
# -0.25|-0.50|90|0 and 0.50|-0.75|270|0.
env = DiscreteEnvironment(**params)
env.init_scene()
print(env.all_objects)
t = input('Choose a target:(no input to free explore)')
t = None if t == '' else t
state, reper = env.reset(
    agent_state=None,
    target_str=t,
    allow_no_target=True,
    calc_best_len=True
)
press_key = None

# 数字都是键码
action_dict = {
    119: 'MoveAhead', 97: 'TurnLeft',
    100: 'TurnRight', 115: 'Done',
    105: 'LookUp', 107: 'LookDown', 120: 'BackOff'
    }
command_dict = {
    116: 'Teleport',
    114: 'reset'
}
reward_sum = 0
step = 0
while True:
    pic = state['RGB'][:]
    print(env.agent_state)
    # fm = state['res18fm'][:]
    # print(torch.tensor(fm[0]))
    # RGB to BGR
    pic = pic[:, :, ::-1]
    # print(fc.shape)
    cv2.imshow("Env", pic)
    press_key = cv2.waitKey(0)
    # print(press_key)
    if press_key in action_dict.keys():
        step += 1
        state, reward, done, info = env.step(action_dict[press_key])
        print('Instant Reward:', reward)
        reward_sum += reward
        if done:
            pic = state['RGB'][:, :, ::-1]
            reward_sum = 0
            ss = info['success']
            print('Total Reward:', reward_sum)
            if ss:
                print('SPL:', info['best_len']/step)
            else:
                print('failed')
            print('Ep len', step)
            step = 0
            print('Env is done. Press anykey to Reset.')
            cv2.imshow("Env", pic)
            cv2.waitKey(0)
            print(env.all_objects)
            t = input('Choose a target:')
            t = None if t == '' else t
            state, reper = env.reset(
                target_str=t, allow_no_target=True, calc_best_len=True
            )
    elif press_key == 27:
        break
    elif press_key == 116:
        p = input('Set State:')
        env.set_agent_state(p, env.all_visible_states)
        state = env.get_obs()
    elif press_key == 114:
        state, reper = env.reset(target_str=None,
                                 allow_no_target=True, calc_best_len=True)
    else:
        print('Unsupported action')
