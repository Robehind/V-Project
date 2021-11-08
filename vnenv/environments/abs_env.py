from typing import Any, Tuple, Dict, Optional
import random
import numpy as np


class AbsObsRender:
    # 用于根据状态返回对应的观察，管理观察的数据类型和大小。字符串
    def __init__(
        self,
        obs_dict: Dict[str, Any],
        map_sz: Tuple
    ) -> None:
        self.obs_dict = obs_dict
        self.all_obs_info = {
            'map': (map_sz, np.dtype(np.int8)),
            'mat': (map_sz, np.dtype(np.int8)),
            'relapos': ((2,), np.dtype(np.int32)),
            'abs_agt': ((2,), np.dtype(np.int32)),
            'abs_tgt': ((2,), np.dtype(np.int32))
        }

    def init(self, scene_id):
        # 切换场景时，可能需要加载不同的文件
        pass

    def data_info(self):
        # 返回所选择的观察的大小和数据类型
        keys = list(self.obs_dict.keys())
        shapes = {
            k: self.all_obs_info[v][0]
            for k, v in self.obs_dict.items()
        }
        dtypes = {
            k: self.all_obs_info[v][1]
            for k, v in self.obs_dict.items()
        }
        return keys, shapes, dtypes

    def render(self, state) -> Dict[str, np.ndarray]:
        # 根据状态求出当前的观察, 数据为numpy矩阵。
        agent_pos = state[0]
        target = state[1]
        map_ = state[2]
        obs = {}
        for k, v in self.obs_dict.items():
            if v == 'map':
                obs[k] = [
                    [1 if y == '#' else 0 for y in x]
                    for x in map_
                ]
                obs[k] = np.array(obs[k], dtype=np.int8)
            if v == 'mat':
                obs[k] = [
                    [1 if y == '#' else 0 for y in x]
                    for x in map_
                ]
                obs[k][agent_pos[0]][agent_pos[1]] += 2
                obs[k][target[0]][target[1]] += 3
                obs[k] = np.array(obs[k], dtype=np.int8)
            if v == 'relapos':
                obs[k] = (agent_pos[0]-target[0], agent_pos[1]-target[1])
                obs[k] = np.array(obs[k], dtype=np.int32)
            if v == 'abs_agt':
                obs[k] = agent_pos
                obs[k] = np.array(obs[k], dtype=np.int32)
            if v == 'abs_tgt':
                obs[k] = target
                obs[k] = np.array(obs[k], dtype=np.int32)
        return obs


class AbsDynamics:
    # 只管理环境内部状态的变化，不涉及事件判断和观察生成，epi是否结束也不归我管，可以说和强化学习毫无关系
    def __init__(
        self,
        action_dict: Dict = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'Done': None
            },
        scene_ids=[0]
    ) -> None:
        self.action_labels = []
        self.actions = []
        for k, v in action_dict.items():
            self.action_labels.append(k)
            self.actions.append(v)
        self.action_sz = len(self.actions)
        self.scene_ids = scene_ids
        self.scene_id = None
        self.map = None
        self.target = None
        self.agent_pos = (-1, -1)
        self.min_len = 0

    def init(
        self,
        map_id: Optional[str] = None
    ):
        # 切换房间时可能加载不同的文件
        # load scene
        if map_id is None:
            map_id = random.choice(self.scene_ids)
        self.scene_id = map_id
        self.map = _maps[map_id]
        self.n = len(self.map)
        self.m = len(self.map[0])
        self.stepable = [
            (i, j)
            for i in range(self.n)
            for j in range(self.m)
            if self.map[i][j] != '#'
        ]
        return map_id

    def get_state(self):
        # 返回一个可以支持render和event进行判断的状态即可
        return [self.agent_pos, self.target, self.map]

    def reset(
        self,
        target: Optional[str] = None,
        agent_pos: Optional[str] = None,
        min_len: bool = False
    ):
        # 重置环境状态
        # place target
        if target is None:
            target = random.choice(self.stepable)
            assert self.map[target[0]][target[1]] != '#'
        self.target = target
        # place agent
        if agent_pos is None:
            agent_pos = random.choice(list(
                set(self.stepable) - set(target)
                )
            )
        assert self.map[agent_pos[0]][agent_pos[1]] != '#'
        self.agent_pos = agent_pos
        # best len
        if min_len:
            self.min_len = self.dfs(agent_pos, target)
        return self.get_state()

    def step(self, act_idx):
        # 以动作的序号来交互
        a = self.actions[act_idx]
        if a is None:
            return self.get_state()
        x, y = self.agent_pos[0]+a[0], self.agent_pos[1]+a[1]
        if self.map[x][y] != '#':
            self.agent_pos = (x, y)
        return self.get_state()

    def dfs(self, st, ed):
        n_node = [st]
        c_node = []
        deep = -1
        vis = [[0]*self.m for _ in range(self.n)]
        vis[st[0]][st[1]] = 1
        while n_node:
            deep += 1
            c_node = n_node
            n_node = []
            for n in c_node:
                for a in self.actions:
                    if a is None:
                        continue
                    x, y = n[0]+a[0], n[1]+a[1]
                    if (x, y) == ed:
                        return deep+1
                    if vis[x][y] != 1 and self.map[x][y] != '#':
                        n_node.append((x, y))
                        vis[x][y] = 1


class AbsEvent:
    # 管理人为定义的特殊事件，包括epi的结束，奖励函数
    def __init__(
        self,
        max_steps,
        reward_func={
            'collision': -0.1,
            'step': -0.01,
            'success': 5,
            'fail': 0
        }
    ) -> None:
        self.last_state = None
        self.rewards = reward_func
        self.steps = 0
        self.max_steps = max_steps

    def reset(self, init_state):
        # 重置事件，把历史事件清空
        self.last_state = init_state
        self.done = False
        self.steps = 0

    def judge(self, state, action):
        # 传入新状态和动作，结合过去状态判断事件
        assert self.done is False
        event = 'step'
        self.steps += 1
        info = {}
        if action == 'Done':
            info['agent_done'] = True
            self.done = True
            event = 'fail'
            info['success'] = False
            if state[0] == state[1]:
                event = 'success'
                info['success'] = True
        elif self.steps >= self.max_steps:
            self.done = True
            event = 'fail'
            info['success'] = False
        elif self.last_state[0] == state[0]:
            event = 'collision'
        self.last_state = state
        info['event'] = event
        return self.rewards[event], self.done, info


class AbsEnv:
    # 从三个角度封装和控制环境的功能
    # 1.动态特性，用参数控制交互方式和状态改变规则，例如设定动作、设定环境动态等
    # 2.观察生成，根据状态产生观察，用参数控制具体什么观察。被预处理的观察也通过这个组件控制
    # 3.事件判定，用参数控制交互过程中的事件以及对应的奖励
    def __init__(
        self,
        dynamics_args,  # 如果动作是可定制的，通过这个参数来控制
        obs_args,
        event_args,
        seed=None,
        train=True
    ) -> None:

        if seed is not None:
            random.seed(seed)
        self.obs_render = AbsObsRender(**obs_args)
        self.dynamics_ctl = AbsDynamics(**dynamics_args)
        self.event_ctl = AbsEvent(**event_args)

        # 足以唯一确定场景观测的量。例如在静态场景中，机器人的位姿就足够确定当前的观测
        self.state = None
        self.his_state = []
        self.info = {}

        # 动作信息，至少包含动作数量和字符串描述
        self.action_sz = self.dynamics_ctl.action_sz
        self.scene_id = None

    @classmethod
    def args_maker(cls, env_args, proc_num):
        # 如果不同进程环境参数不同，用该方法封装
        # 最简单的情况下，复制n次, seed递增
        out = [env_args.copy() for _ in range(proc_num)]
        if 'seed' in env_args:
            for i, e in enumerate(out):
                e['seed'] += i
        return out

    def re_seed(self, seed):
        # warning: use it when env has its own proc
        random.seed(seed)

    def update_settings(self, settings):
        raise NotImplementedError

    def export_settings(self):
        raise NotImplementedError

    def init_scene(self, scene_id=None):
        # 加载某一场景的资源和信息，不涉及任何智能体的初始化
        if scene_id != self.scene_id or self.scene_id is None:
            self.scene_id = self.dynamics_ctl.init(scene_id)
            self.obs_render.init(self.scene_id)
        self.his_state = []
        self.state = None

    def init_task(self, *args, **kwargs):
        # 初始化目标，初始化智能体位置等，即初始化环境的状态
        self.state = self.dynamics_ctl.reset(*args, **kwargs)
        self.event_ctl.reset(self.state)

    def reset(self, *args, **kwargs):
        # 重置环境
        scene_id = None
        if 'scene_id' in kwargs:
            scene_id = kwargs.pop('scene_id')
        self.init_scene(scene_id)
        self.init_task(*args, **kwargs)
        self.info = {
            'scene_id': self.scene_id,
            'min_len': self.dynamics_ctl.min_len
        }
        return self.get_obs()

    def str_step(self, action: str):
        action = self.dynamics_ctl.action_labels.index(action)
        return self.step(action)

    def get_obs(self):
        return self.obs_render.render(self.state)

    def step(self, action: int):
        self.state = self.dynamics_ctl.step(action)
        action_label = self.dynamics_ctl.action_labels[action]
        self.his_state.append(self.state)
        r, d, info = self.event_ctl.judge(self.state, action_label)
        self.info.update(info)
        return self.get_obs(), r, d, self.info

    def data_info(self) -> Tuple[list, dict, dict]:
        return self.obs_render.data_info()

    def close(self):
        pass


_maps = [
    [
        "####################",
        "#   #          #   #",
        "#   ####    #  # ###",
        "#   #    #  #  #   #",
        "## ##  ########### #",
        "#  #   # #  #      #",
        "#    ### #     #####",
        "###      #  #  #   #",
        "#    ##     ##   # #",
        "####################",
    ],
    [
        "####################",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "#                  #",
        "####################",
    ],
    [
        "#####",
        "#   #",
        "#   #",
        "#   #",
        "#####",
    ],
    [
        "###",
        "# #",
        "# #",
        "###"
    ]
]

if __name__ == '__main__':
    # human explore test
    def vis(mat):
        for row in mat:
            pp = ''
            for s in row:
                if s == 0:
                    pp += ' '
                if s == 1:
                    pp += '#'
                if s == 2:
                    pp += 'A'
                if s == 3:
                    pp += 'T'
                if s == 5:
                    pp += '@'
            print(pp)
    # params
    d_args = {'scene_ids': [2]}
    o_args = {
        'obs_dict': {
            'vis': 'mat',
            'agt_pos': 'abs_agt',
            'tgt_pos': 'abs_tgt',
            'rela': 'relapos'
        },
        'map_sz': (5, 5)
    }
    e_args = {'max_steps': 10}
    env = AbsEnv(d_args, o_args, e_args)

    # 数字都是键码
    action_dict = {
        'w': 'up', 'a': 'left',
        'd': 'right', 's': 'Done',
        'x': 'down'
        }
    reward_sum = 0
    step = 0
    obs = env.reset(min_len=True)
    print(env.data_info())
    while True:
        vis(obs.pop('vis'))
        print(obs)
        press_key = input()
        if press_key in action_dict.keys():
            step += 1
            obs, reward, done, info = env.str_step(action_dict[press_key])
            print('Instant Reward:', reward)
            reward_sum += reward
            if done:
                ss = (info['event'] == 'success')
                print('Total Reward:', reward_sum)

                reward_sum = 0
                if ss:
                    print('SPL:', (info['min_len']+1)/step)
                else:
                    print('failed')
                print('Ep len', step)
                step = 0
                done_type = 'agent done' if 'agent_done' in info \
                    else 'max step'
                print(f'Env is done by {done_type}')
                obs = env.reset(min_len=True)
        elif press_key == 'quit':
            break
        elif press_key == 'r':
            obs, init_info = env.reset(min_len=True)
            print('reseted')
            print(init_info)
        else:
            print('Unsupported action')
