from methods.learners.returns_calc import _basic_return, _GAE
import numpy as np


def test_basic_return():
    v_array = np.array([4, 5, 6, 7, 9, 2]).reshape(-1, 1)
    rew = np.array([1, 1, 2, 2, 3]).reshape(-1, 1)
    mask = np.array([1, 1, 0, 1, 1]).reshape(-1, 1)
    r1 = _basic_return(v_array, rew, mask, gamma=0.5, nsteps=float("inf"))
    r1_t = np.array([2, 2, 2, 4, 4]).reshape(-1, 1)
    assert np.allclose(r1, r1_t)
    r2 = _basic_return(v_array, rew, mask, gamma=1, nsteps=1)
    r2_t = np.array([6, 7, 2, 11, 5]).reshape(-1, 1)
    assert np.allclose(r2, r2_t)
    r2 = _basic_return(v_array, rew, mask, gamma=1, nsteps=3)
    r2_t = np.array([4, 3, 2, 7, 5]).reshape(-1, 1)
    assert np.allclose(r2, r2_t)
    # test for batch calc
    v_array = np.array([[4, 4], [5, 5], [6, 6], [7, 7], [9, 9], [2, 2]])
    rew = np.array([[1, 1], [1, 1], [2, 2], [2, 2], [3, 3]])
    mask = np.array([[1, 1], [1, 1], [0, 0], [1, 1], [1, 1]])
    r1 = _basic_return(v_array, rew, mask, gamma=0.5, nsteps=float("inf"))
    r1_t = [[x, x] for x in [2, 2, 2, 4, 4]]
    assert np.allclose(r1, np.array(r1_t).reshape(-1, 1))
    r2 = _basic_return(v_array, rew, mask, gamma=1, nsteps=3)
    r2_t = [[x, x] for x in [4, 3, 2, 7, 5]]
    assert np.allclose(r2, np.array(r2_t).reshape(-1, 1))


def test_gae():
    v_array = np.array([[4, 4], [5, 5], [6, 6], [7, 7], [9, 9], [2, 2]])
    rew = np.array([[1, 1], [1, 1], [2, 2], [2, 2], [3, 3]], dtype=np.float32)
    mask = np.array([[1, 1], [1, 1], [0, 0], [1, 1], [1, 1]])
    adv = _GAE(v_array, rew, mask, gamma=0.5, lbd=1)
    returns = [[x, x] for x in [2, 2, 2, 4, 4]]
    returns = np.array(returns).reshape(-1, 1)
    assert np.allclose(adv, returns - v_array[:-1].reshape(-1, 1))
    adv = _GAE(v_array, rew, mask, gamma=1, lbd=0)
    returns = [[x, x] for x in [6, 7, 2, 11, 5]]
    returns = np.array(returns).reshape(-1, 1)
    assert np.allclose(adv, returns - v_array[:-1].reshape(-1, 1))
    adv = _GAE(v_array, rew, mask, gamma=1, lbd=0.5)
    returns = [[x, x] for x in [2, 0, -4, 2, -4]]
    returns = np.array(returns).reshape(-1, 1)
    assert np.allclose(adv, returns)
