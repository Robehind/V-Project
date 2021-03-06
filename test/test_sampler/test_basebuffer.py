from methods.samplers import BaseBuffer
import numpy as np
from collections import deque
import pytest
from gym.spaces import Dict, Box
INF = float("inf")


def cross_cmp(a, b, st=0, ed=None):
    # assert a.shape[1] == len(b), f'{a.shape[1]} vs {len(b)}'
    if ed is None:
        ed = len(b[0])
    assert a.shape[0] == ed-st, f'{a.shape[0]} vs {len(b[0])}'
    n = a.shape[1]
    for i in range(n):
        x = a[:, i]
        flag = False
        for y in b:
            if np.allclose(x, y[st:ed]):
                flag = True
                break
        if not flag:
            return False
    return True


@pytest.mark.parametrize(
    ["env_num", "exp_length", "sample_num", "max_exp_num", "rounds"],
    [
        [4, 3, 4, 4, 1],
        [3, 3, 4, 4, 2],
        [1, 3, 4, 5, 7],
        [3, 3, 3, 6, 1]
    ]
)
def test_buffer(
    env_num,
    exp_length,
    sample_num,
    max_exp_num,
    rounds
):
    obs_spc = Dict({
        'map': Box(-INF, INF, (4, 4, 7), np.float32),
        'fc': Box(-INF, INF, (1, 10), np.float32),
    })
    bb = BaseBuffer(
        obs_spc,
        {'lstm': (10, )},
        {'lstm': np.float32},
        exp_length=exp_length,
        sample_num=sample_num,
        env_num=env_num,
        max_exp_num=max_exp_num
    )
    mapp = deque(maxlen=max_exp_num)
    lstm = deque(maxlen=max_exp_num)
    r = deque(maxlen=max_exp_num)
    m = deque(maxlen=max_exp_num)
    a = deque(maxlen=max_exp_num)
    fc = deque(maxlen=max_exp_num)
    for _ in range(rounds):
        for _ in range(env_num):
            mapp.append(np.random.rand(exp_length, 4, 4, 7))
            fc.append(np.random.rand(exp_length, 1, 10))
            lstm.append(np.random.rand(exp_length, 10))
            r.append(np.random.randint(0, 10, (exp_length, )))
            m.append(np.random.randint(0, 2, (exp_length, )))
            a.append(np.random.randint(0, 5, (exp_length, )))
        for i in range(exp_length):
            write_in_data = {
                'obs': {
                    'map': np.array(list(mapp)[-env_num:])[:, i],
                    'fc': np.array(list(fc)[-env_num:])[:, i],
                },
                'rct': {'lstm': np.array(list(lstm)[-env_num:])[:, i]},
                'r': np.array(list(r)[-env_num:])[:, i],
                'm': np.array(list(m)[-env_num:])[:, i],
                'a': np.array(list(a)[-env_num:])[:, i]
            }
            bb.write_in(
                **write_in_data
            )
    out = bb.sample()
    assert cross_cmp(out['r'], r)
    assert cross_cmp(out['a'], a)
    assert cross_cmp(out['m'], m)
    assert cross_cmp(out['obs']['map'], mapp)
    assert cross_cmp(out['rct']['lstm'], lstm)
    assert cross_cmp(out['obs']['fc'], fc)
    # test for range specifiction
    out = bb.sample(st=1, ed=3)
    assert cross_cmp(out['r'], r, 1, 3)
    assert cross_cmp(out['a'], a, 1, 3)
    assert cross_cmp(out['m'], m, 1, 3)
    assert cross_cmp(out['obs']['map'], mapp, 1, 3)
    assert cross_cmp(out['rct']['lstm'], lstm, 1, 3)
    assert cross_cmp(out['obs']['fc'], fc, 1, 3)
