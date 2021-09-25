import multiprocessing as mp
from typing import Dict, List
import numpy as np
import ctypes


# borrowed from https://github.com/openai/baselines, modified
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents
    (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


_NP_TO_CT = {np.dtype(np.float32): ctypes.c_float,
             np.dtype(np.int32): ctypes.c_int32,
             np.dtype(np.int8): ctypes.c_int8,
             np.dtype(np.uint8): ctypes.c_char,
             np.dtype(np.bool): ctypes.c_bool}


def make_envs(env_args: Dict, env_class):
    """预封装环境的生成函数，为生成多线程环境服务的"""
    def _env_func():
        return env_class(**env_args)
    return _env_func


class VecEnv:
    closed = False

    def __init__(
        self,
        env_fns: List[callable],
        context: str = 'spawn',
        min_len: bool = False,
        no_sche_no_op: bool = False
    ):
        """
        多线程环境。对env的一个封装，输入的是环境的构造函数.min_len=True下会及算最短路，很慢
        """
        ctx = mp.get_context(context)
        self.env_num = len(env_fns)
        # 随便初始化一个环境，获得关于字段名、数据size和type信息
        env = env_fns[0]()
        self.keys, self.shapes, self.dtypes = env.data_info()
        self.action_sz = env.action_sz
        env.close()
        del env
        # 所有环境共享同一个sche
        self.sche = mp.Manager().list()
        # 按字段构造数据缓存区
        self.data_bufs = [
            {
                k: ctx.Array(
                    _NP_TO_CT[self.dtypes[k]], int(np.prod(self.shapes[k]))
                )
                for k in self.keys
            }
            for _ in env_fns
        ]
        # 生成环境进程
        self.parent_pipes = []
        self.procs = []
        for env_fn, data_buf in zip(env_fns, self.data_bufs):
            parent_pipe, child_pipe = ctx.Pipe()
            env_fn = CloudpickleWrapper(env_fn)
            proc = ctx.Process(
                target=_subproc_worker,
                args=(
                    child_pipe,
                    parent_pipe,
                    env_fn,
                    data_buf,
                    self.shapes,
                    self.dtypes,
                    min_len,
                    self.sche,
                    no_sche_no_op
                )
            )
            proc.daemon = True
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()
        self.waiting_step = False

    def sche_update(self, sche, clear=False):
        if sche == [] or sche is None:
            return
        if self.waiting_step:
            self.step_wait()
        if clear:
            self.sche.clear()
        self.sche.extend(sche)

    def reset(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        [pipe.recv() for pipe in self.parent_pipes]
        return self.get_obs()

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        _, rews, dones, info = zip(*outs)
        return self.get_obs(), np.array(rews), \
            np.array(dones, dtype=np.dtype(np.int8)), info

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def get_obs(self):
        """从buff里读出obs"""
        result = {}
        for k in self.keys:

            bufs = [b[k] for b in self.data_bufs]
            o = [
                np.frombuffer(
                    b.get_obj(), dtype=self.dtypes[k]
                ).reshape(self.shapes[k])
                for b in bufs
            ]
            result[k] = np.array(o)
        return result


def _subproc_worker(
    pipe,
    parent_pipe,
    env_fn,
    bufs,
    obs_shapes,
    obs_dtypes,
    min_len,
    sche,
    no_op=False  # 在sche为空时，如果为true，则会无任何操作
):
    """
    Control a single environment instance using IPC and
    shared memory.
    """

    def _write_bufs(dict_data):
        # TODO 未改变的数据(例如目标的数据)可以不写入
        for k in dict_data:
            dst = bufs[k].get_obj()
            dst_np = np.frombuffer(
                dst, dtype=obs_dtypes[k]
            ).reshape(obs_shapes[k])  # pylint: disable=W0212
            np.copyto(dst_np, dict_data[k])
    env = env_fn.x()
    parent_pipe.close()
    waiting = False
    try:
        while True:
            cmd, data = pipe.recv()
            if waiting and cmd != 'close':
                pipe.send((None, 0, 0, None))
                continue
            if cmd == 'reset':
                try:
                    ss = sche.pop(0)
                    obs = env.reset(
                        **ss, min_len=min_len
                    )
                except IndexError:
                    obs = env.reset(min_len=min_len)
                    waiting = no_op
                pipe.send((_write_bufs(obs)))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    if len(sche) == 0:
                        obs = env.reset(min_len=min_len)
                        waiting = no_op
                    else:
                        obs = env.reset(
                            **sche.pop(0), min_len=min_len
                        )
                pipe.send((_write_bufs(obs), reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render())
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('VecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()
