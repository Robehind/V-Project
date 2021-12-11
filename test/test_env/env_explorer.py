from vnenv.environments import DiscreteEnvironment
import cv2
params = dict(
    event_args={
        "reward_dict": {
            "collision": -0.1,
            "step": -0.01,
            "success": 10.0,
            "fail": 0,
        },
        'max_steps': 100,
    },
    dynamics_args={
        'offline_data_dir': '../vdata/thordata',
        'action_dict': {
            'MoveAhead': ['m0'],
            'TurnLeft': ['r-45'],
            'TurnRight': ['r45'],
            'LookUp': ['p-30'],
            'LookDown': ['p30'],
            'Done': None
        },
        'rotate_angle': 45,
        'move_angle': 45,
        'horizon_angle': 30,
    },
    obs_args={
        "obs_dict": {
            "fc": "resnet50fc_no_norm.hdf5",
            "RGB": "images.hdf5",
        },
        'target_dict': {
            'glove': '../vdata/word_embedding/word_embedding.hdf5',
            'img': 'images.hdf5'
        },
        'info_scene': "FloorPlan1_physics"
    },
)
# -0.25|-0.50|90|0 and 0.50|-0.75|270|0.
env = DiscreteEnvironment(**params)
env.update_settings(dict(chosen_scenes={'kitchen': '9'}))
env.init_scene()
print(env.all_objects)
t = input('Choose a target:(no input to free explore)')
t = None if t == '' else t
state = env.reset(
    agent_state='-1.25|-0.75|45|0',
    target_str=t,
    allow_no_target=True,
    min_len=True
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
t_img = state['img'][:, :, ::-1]
cv2.imshow("tgt", t_img)
while True:
    pic = state['RGB'][:]
    print(env.agent_state)
    # RGB to BGR
    pic = pic[:, :, ::-1]
    cv2.imshow("Env", pic)
    press_key = cv2.waitKey(0)
    if press_key in action_dict.keys():
        step += 1
        state, reward, done, info = env.str_step(action_dict[press_key])
        print('Instant Reward:', reward)
        reward_sum += reward
        if done:
            pic = state['RGB'][:, :, ::-1]
            reward_sum = 0
            ss = info['success']
            print('Total Reward:', reward_sum)
            if ss:
                print('SPL:', info['min_len']/step)
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
            state = env.reset(
                target_str=t, allow_no_target=True, min_len=True
            )
            t_img = state['img'][:, :, ::-1]
            cv2.imshow("tgt", t_img)
    elif press_key == 27:
        break
    elif press_key == 116:
        p = input('Set State:')
        env.set_agent_state(p, env.all_visible_states)
        state = env.get_obs()
    elif press_key == 114:
        state = env.reset(target_str=None,
                          allow_no_target=True, min_len=True)
    else:
        print('Unsupported action')
