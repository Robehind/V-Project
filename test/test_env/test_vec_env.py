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
            },
            'map_sz': (10, 20)
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
             'event': 'success', 'agent_done': True, 'success': True},
            {'scene_id': 1, 'min_len': 4, 'event': 'step'},
            {'scene_id': 0, 'min_len': 3,
             'event': 'success', 'agent_done': True, 'success': True},
            {'scene_id': 1, 'min_len': 1, 'event': 'fail',
             'agent_done': True, 'success': False},
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
    idx = [0, 1, 2, 3]
    n_idx = []
    for a in Venv.reset()['rela']:
        flag = False
        for i in idx:
            if np.allclose(a, obss[0]['rela'][i]):
                flag = True
                n_idx.append(i)
                idx.remove(i)
                break
        assert flag
    # assert np.allclose(obss[0]['rela'], Venv.reset()['rela'])
    for i, a in enumerate(actions):
        obs, r, d, info = Venv.step(np.array(a)[n_idx])
        assert np.allclose(r, rs[i][n_idx])
        assert np.allclose(d, ds[i][n_idx])
        p = 0
        for ii in info:
            assert ii == infos[i][n_idx[p]]
            p += 1
        if i == 4:
            assert np.allclose(obs['rela'][n_idx.index(1)],
                               obss[i+1]['rela'][1])
        else:
            assert np.allclose(obs['rela'], obss[i+1]['rela'][n_idx])
    Venv.close()
    # test for proc waiting feature
    Venv = VecEnv(env_fns, min_len=True, no_sche_no_op=True)
    Venv.sche_update([sche[0]])
    init_obs = Venv.reset()['rela']
    obs, _, _, _ = Venv.step([actions[0][0] for _ in range(4)])
    for i in range(4):
        if not np.allclose(obs['rela'][i], init_obs[i]):
            proc = i
    other = [0, 1, 2, 3]
    other.remove(proc)
    for i, a in enumerate(actions[1:]):
        obs, r, d, info = Venv.step([a[0] for _ in range(4)])
        if i != 3:
            assert np.allclose(obs['rela'][proc], obss[i+2]['rela'][0])
        assert np.allclose(obs['rela'][other], init_obs[other])
        assert np.allclose(r[proc], rs[i+1][0])
        assert np.allclose(r[other], 0)
        assert np.allclose(d[proc], ds[i+1][0])
        assert np.allclose(d[other], 0)
        assert info[proc] == infos[i+1][0]
        for j in other:
            assert info[j] is None
    Venv.close()


if __name__ == '__main__':
    test_vec_env()
