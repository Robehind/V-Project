from vnenv.samplers import BaseBuffer
import numpy as np


def test_buffer():
    np.random.seed(1114)
    bb = BaseBuffer(
        {'map': (4, 4, 7), 'fc': (1, 10)},
        {'map': np.float32, 'fc': np.float},
        4,
        3
    )
    mapp, fc, r, m, a = [], [], [], [], []
    for _ in range(3):
        mapp.append(np.random.rand(4, 4, 4, 7))
        fc.append(np.random.rand(4, 1, 10))
        r.append(np.random.randint(0, 10, (4,)))
        m.append(np.random.randint(0, 2, (4,)))
        a.append(np.random.randint(0, 5, (4,)))
        write_in_data = {
            'o': {
                'map': mapp[-1],
                'fc': fc[-1],
            },
            'r': r[-1],
            'm': m[-1],
            'a': a[-1]
        }
        bb.write_in(
            **write_in_data
        )
    mapp.append(np.random.rand(4, 4, 4, 7))
    fc.append(np.random.rand(4, 1, 10))
    bb.one_more_obs({
        'map': mapp[-1],
        'fc': fc[-1],
    })
    out = bb.batched_out()
    assert np.allclose(out['o']['map'], np.array(mapp).reshape(-1, 4, 4, 7))
    assert np.allclose(out['o']['fc'], np.array(fc).reshape(-1, 1, 10))
    assert np.allclose(out['r'], np.array(r))
    assert np.allclose(out['a'], np.array(a).reshape(-1, 1))
    assert np.allclose(out['m'], np.array(m))


if __name__ == '__main__':
    test_buffer()
