from vnenv.samplers import BaseSampler
import numpy as np
import torch
import pytest


# TODO not well tested
def cross_cmp(a, b):
    # assert a.shape[1] == len(b), f'{a.shape[1]} vs {len(b)}'
    assert a.shape == b.shape
    n = a.shape[1]
    for i in range(n):
        x = a[:, i]
        flag = False
        for j in range(n):
            if np.allclose(x, b[:, j]):
                flag = True
                break
        if not flag:
            return False
    return True


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
    def __init__(self, rct_on=True) -> None:
        self.step = -1
        self.acts = _acts
        self.rct_shapes = {}
        self.rct_dtypes = {}
        self.rct_on = rct_on
        if rct_on:
            self.rct_shapes = {'lstm': (1, )}
            self.rct_dtypes = {'lstm': torch.float32}
            self.rcts = torch.randn(5, 4, 1)
            self.rcts[0] = 0

    def action(self, obs):
        self.step += 1
        return np.array(self.acts[self.step])

    def reset_rct(self, idx):
        pass

    def get_rct(self):
        return {'lstm': self.rcts[self.step+1]} if self.rct_on else {}

    def close(self):
        pass


@pytest.mark.parametrize('rct_on', [True, False])
def test_sampler(rct_on):
    cl = TESTcl()
    Venv = TESTenv()
    agent = TESTagent(rct_on)
    sampler = BaseSampler(
        Venv,
        agent,
        cl,
        batch_size=20,
        exp_length=5,
        buffer_limit=4
    )
    out = sampler.sample()
    assert cross_cmp(out['obs']['rela'], np.array(_obss))
    assert cross_cmp(out['r'], np.array(_rs))
    assert cross_cmp(out['a'], np.array(_acts))
    assert cross_cmp(out['m'], np.array(1 - _ds))
    if rct_on:
        assert cross_cmp(out['rct']['lstm'], agent.rcts)
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
