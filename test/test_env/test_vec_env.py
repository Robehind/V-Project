from vnenv.environments.abs_env import AbsEnv
from vnenv.environments.env_wrapper import VecEnv, make_envs
import numpy as np


def test_vec_env():
    # params
    env_args = dict(
        dynamics_args={
            'scene_ids': [0, 1],
            'action_dict': {
                'up': (-1, 0),
                'down': (1, 0),
                'left': (0, -1),
                'right': (0, 1),
                'Done': None
            }
        },
        obs_args={
            'obs_dict': {
                'vis': 'mat',
                'rela': 'relapos'
            }
        },
        event_args={'max_steps': 50},
        seed=1114
    )
    sche = [
        {'scene_id': 0, 'target': (2, 8), 'agent_pos': (1, 5)},
        {'scene_id': 1, 'target': (3, 3), 'agent_pos': (5, 5)},
        {'scene_id': 0, 'target': (8, 18), 'agent_pos': (7, 16)},
        {'scene_id': 1, 'target': (5, 10), 'agent_pos': (6, 10)},
    ]
    rs = [
        [-0.01, -0.01, -0.1, -0.01],
        [-0.01, -0.01, -0.01, -0.01],
        [-0.01, -0.01, -0.01, -0.01],
        [-0.01, -0.01, -0.01, -0.01],
        [5, -0.01, 5, 0]
    ]
    rs = np.array(rs)
    ds = np.zeros((5, 4))
    ds[4, 0] = 1
    ds[4, 2] = 1
    ds[4, 3] = 1
    infos = [
        (
            {'scene_id': 0, 'min_len': 4, 'event': 'step'},
            {'scene_id': 1, 'min_len': 4, 'event': 'step'},
            {'scene_id': 0, 'min_len': 3, 'event': 'collision'},
            {'scene_id': 1, 'min_len': 1, 'event': 'step'},
        ),
        (
            {'scene_id': 0, 'min_len': 4, 'event': 'step'},
            {'scene_id': 1, 'min_len': 4, 'event': 'step'},
            {'scene_id': 0, 'min_len': 3, 'event': 'step'},
            {'scene_id': 1, 'min_len': 1, 'event': 'step'},
        ),
        (
            {'scene_id': 0, 'min_len': 4, 'event': 'step'},
            {'scene_id': 1, 'min_len': 4, 'event': 'step'},
            {'scene_id': 0, 'min_len': 3, 'event': 'step'},
            {'scene_id': 1, 'min_len': 1, 'event': 'step'},
        ),
        (
            {'scene_id': 0, 'min_len': 4, 'event': 'step'},
            {'scene_id': 1, 'min_len': 4, 'event': 'step'},
            {'scene_id': 0, 'min_len': 3, 'event': 'step'},
            {'scene_id': 1, 'min_len': 1, 'event': 'step'},
        ),
        (
            {'scene_id': 0, 'min_len': 4,
             'event': 'success', 'agent_done': True},
            {'scene_id': 1, 'min_len': 4, 'event': 'step'},
            {'scene_id': 0, 'min_len': 3,
             'event': 'success', 'agent_done': True},
            {'scene_id': 1, 'min_len': 1, 'event': 'fail', 'agent_done': True},
        ),

    ]
    actions = [
        [3, 2, 0, 1],
        [3, 2, 3, 0],
        [3, 3, 3, 1],
        [1, 3, 1, 0],
        [4, 3, 4, 4]
    ]
    obss = [
        {'rela': np.array([[-1, -3], [2, 2], [-1, -2], [1, 0]])},
        {'rela': np.array([[-1, -2], [2, 1], [-1, -2], [2, 0]])},
        {'rela': np.array([[-1, -1], [2, 0], [-1, -1], [1, 0]])},
        {'rela': np.array([[-1, 0], [2, 1], [-1, 0], [2, 0]])},
        {'rela': np.array([[0, 0], [2, 2], [0, 0], [1, 0]])},
        {'rela': np.array([[-3, 10], [2, 3], [0, 8], [-7, -11]])},
    ]
    env_args_list = AbsEnv.args_maker(env_args, 4)
    env_fns = [make_envs(e, AbsEnv) for e in env_args_list]
    Venv = VecEnv(env_fns, min_len=True)
    Venv.sche_update(sche)
    assert np.allclose(obss.pop(0)['rela'], Venv.reset()['rela'])
    for i, a in enumerate(actions):
        obs, r, d, info = Venv.step(a)
        assert np.allclose(obs['rela'], obss[i]['rela'])
        assert np.allclose(r, rs[i])
        assert np.allclose(d, ds[i])
        assert info == infos[i]


if __name__ == '__main__':
    test_vec_env()
