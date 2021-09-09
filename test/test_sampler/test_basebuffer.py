from vnenv.samplers import BaseBuffer
import numpy as np
from collections import deque


def cross_cmp(a, b):
    # assert a.shape[1] == len(b), f'{a.shape[1]} vs {len(b)}'
    assert a.shape[0] == len(b[0]), f'{a.shape[0]} vs {len(b[0])}'
    n = a.shape[1]
    for i in range(n):
        x = a[:, i]
        flag = False
        for y in b:
            if np.allclose(x, y):
                flag = True
                break
        if not flag:
            return False
    return True


def test_buffer(
    env_num=4,
    exp_length=3,
    sample_num=4,
    max_exp_num=4,
    rounds=1
):
    np.random.seed(1114)
    bb = BaseBuffer(
        {'map': (4, 4, 7), 'fc': (1, 10)},
        {'map': np.float32, 'fc': np.float},
        exp_length=exp_length,
        sample_num=sample_num,
        env_num=env_num,
        max_exp_num=max_exp_num
    )
    mapp = deque(maxlen=max_exp_num)
    r = deque(maxlen=max_exp_num)
    m = deque(maxlen=max_exp_num)
    a = deque(maxlen=max_exp_num)
    fc = deque(maxlen=max_exp_num)
    for _ in range(rounds):
        for _ in range(env_num):
            mapp.append(np.random.rand(exp_length+1, 4, 4, 7))
            fc.append(np.random.rand(exp_length+1, 1, 10))
            r.append(np.random.randint(0, 10, (exp_length, )))
            m.append(np.random.randint(0, 2, (exp_length, )))
            a.append(np.random.randint(0, 5, (exp_length, )))
        for i in range(exp_length):
            write_in_data = {
                'o': {
                    'map': np.array(list(mapp)[-env_num:])[:, i],
                    'fc': np.array(list(fc)[-env_num:])[:, i],
                },
                'r': np.array(list(r)[-env_num:])[:, i],
                'm': np.array(list(m)[-env_num:])[:, i],
                'a': np.array(list(a)[-env_num:])[:, i]
            }
            bb.write_in(
                **write_in_data
            )
        bb.one_more_obs({
            'map': np.array(list(mapp)[-env_num:])[:, -1],
            'fc': np.array(list(fc)[-env_num:])[:, -1],
        })
    out = bb.sample()
    assert cross_cmp(out['r'], r)
    assert cross_cmp(out['a'].reshape(-1, sample_num), a)
    assert cross_cmp(out['m'], m)
    assert cross_cmp(out['o']['map'].reshape(-1, sample_num, 4, 4, 7),
                     np.array(mapp))
    assert cross_cmp(out['o']['fc'].reshape(-1, sample_num, 1, 10), fc)


if __name__ == '__main__':
    # case1 env_num == sample_num == max_exp_num
    test_buffer(4, 3, 4, 4, 1)
    # case2 env_num < sample_num == max_exp_num, 2 rounds
    test_buffer(3, 3, 4, 4, 2)
    # case3 env_num < sample_num < max_exp_num, 7 rounds
    test_buffer(1, 3, 4, 5, 7)
    # case4 env_num = sample_num < max_exp_num, 1 rounds
    test_buffer(3, 3, 3, 6, 1)
