from vnenv.samplers import BaseSampler
import numpy as np


class TESTcl:
    def __init__(self) -> None:
        pass

    def next_sche(self, *args, **kwargs):
        pass

    def init_sche(self, *args, **kwargs):
        pass


class TESTenv:
    def __init__(self) -> None:
        self.env_num = 4
        self.dtypes = {'rela': np.int32}
        self.shapes = {'rela': (2,)}
        self.obss = _obss
        self.infos = _infos
        self.rs = _rs
        self.ds = _ds
        self.count = -1

    def reset(self):
        return dict(rela=np.array(self.obss[self.count+1]))

    def step(self, action):
        self.count += 1
        return dict(rela=np.array(self.obss[self.count+1])), \
            self.rs[self.count], self.ds[self.count], self.infos[self.count]

    def close(self):
        pass


class TESTagent:
    def __init__(self) -> None:
        self.step = -1
        self.acts = _acts

    def action(self, obs, done):
        self.step += 1
        return np.array(self.acts[self.step])

    def clear_mems(self):
        pass

    def close(self):
        pass


# TODO 跑通就算成功...
def test_sampler():
    cl = TESTcl()
    Venv = TESTenv()
    agent = TESTagent()
    sampler = BaseSampler(Venv, agent, cl, 20)
    out = sampler.run()
    assert np.allclose(out['o']['rela'], np.array(_obss).reshape(-1, 2))
    assert np.allclose(out['r'], np.array(_rs))
    assert np.allclose(out['a'], np.array(_acts).reshape(-1, 1))
    assert np.allclose(out['m'], np.array(1 - _ds))
    records = sampler.pop_records()
    assert records['epis'] == 3
    assert records['success_rate'] == 2/3
    assert np.allclose(records['total_reward'], 9.79/3)
    assert records['total_steps'] == 5
    sampler.close()


_acts = [
    [3, 2, 0, 1],
    [3, 2, 3, 0],
    [3, 3, 3, 1],
    [1, 3, 1, 0],
    [4, 3, 4, 4]
]
_obss = [
    [[-1, -3], [2, 2], [-1, -2], [1, 0]],
    [[-1, -2], [2, 1], [-1, -2], [2, 0]],
    [[-1, -1], [2, 0], [-1, -1], [1, 0]],
    [[-1, 0], [2, 1], [-1, 0], [2, 0]],
    [[0, 0], [2, 2], [0, 0], [1, 0]],
    [[-3, 10], [2, 3], [0, 8], [-7, -11]],
]
_infos = [
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
_rs = [
    [-0.01, -0.01, -0.1, -0.01],
    [-0.01, -0.01, -0.01, -0.01],
    [-0.01, -0.01, -0.01, -0.01],
    [-0.01, -0.01, -0.01, -0.01],
    [5, -0.01, 5, 0]
]
_rs = np.array(_rs)
_ds = np.zeros((5, 4), dtype=np.int8)
_ds[4, 0] = 1
_ds[4, 2] = 1
_ds[4, 3] = 1
if __name__ == '__main__':
    test_sampler()
