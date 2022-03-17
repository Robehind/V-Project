from collections import defaultdict
from functools import lru_cache
from typing import Dict, Tuple, Set, List
import random
from copy import deepcopy
from gym.spaces import Discrete
import numpy as np
from taskenvs import TaskEnv
from .offline_ctrler import OfflineThorCtrler as OCer
from .utils import get_scene_names, get_type


class BaseTargetDrivenTHOR(TaskEnv):
    # 除了观测没有定义，其他都实现好了的一个目标驱动导航环境
    def __init__(
        self,
        actions: List[str],
        rotate_angle: int,
        max_steps: int,
        reward_dict: Dict[str, float],
        ctl_data_dir: str = '../vdata/thordata'
    ) -> None:
        super().__init__()
        camera_up_down = ('LookUp' in actions or 'LookDown' in actions)
        self.ctrler = OCer(
            rotate_angle=rotate_angle,
            camera_up_down=camera_up_down,
            data_dir=ctl_data_dir)
        self.action_space = Discrete(len(actions))
        self.actions = actions
        self.scene = None

        self.get_shortest = False
        self.auto_done = 'Done' not in actions
        self.reward_dict = reward_dict
        self.max_steps = max_steps
        self.steps = 0

        # tasks
        self.all_scenes_by_type = {
            'kitchen': '1-30', 'living_room': '1-30',
            'bedroom': '1-30', 'bathroom': '1-30'}
        self.targets_by_type = defaultdict(set)

    def seed(self, sd):
        random.seed(sd)
        np.random.seed(sd)

    def target_visible(self):
        return self.is_visible(self.scene, self.target, str(self.state))

    @lru_cache(None)
    def is_visible(self, scene: str, target: str, state: str):
        return state in self.ctrler.visible_states(scene, target)

    @lru_cache(None)
    def get_objectIDs(self, scene) -> Set:
        return self.ctrler.objectIDs(scene)

    def get_objects(self, scene) -> Set:
        return {x.split('|')[0] for x in self.get_objectIDs(scene)}

    def get_target_obs(self):
        raise NotImplementedError

    def get_main_obs(self):
        raise NotImplementedError

    def get_observation(self):
        obs = self.get_main_obs()
        obs.update(self.get_target_obs())
        return obs

    def init_obs(self, scene):
        raise NotImplementedError

    def set_tasks(self, t):
        if 'scenes' not in t:
            # 无对应关键字视为全集
            scenes_by_type = self.all_scenes_by_type
        else:
            if isinstance(t['scenes'], dict):
                scenes_by_type = t['scenes']
            else:
                scenes_by_type = None
                self.scenes = deepcopy(t['scenes'])
        if scenes_by_type is not None:
            self.scenes = get_scene_names(scenes_by_type)
        self.ctrler.preload_scenes(self.scenes)
        if 'targets' not in t:
            # 无对应关键字视为全集
            for s in self.scenes:
                self.targets_by_type[get_type(s)] |= self.get_objects(s)
        else:
            self.targets_by_type = {
                k: set(v) for k, v in t['targets'].items()}
        self._tasks = {
            'scenes': deepcopy(self.scenes),
            'targets': deepcopy(self.targets_by_type)}

    def reset(self):
        # choose scene
        while 1:
            scene = random.choice(self.scenes)
            tgts = self.get_objects(scene) & \
                self.targets_by_type[get_type(scene)]
            if tgts == set():
                print(f"Warning: no targtets to choose in {scene}")
                idx = self.scenes.index(scene)
                self.scenes.pop(idx)
                continue
            self.target = random.choice(list(tgts))
            break
        # init scene obs readers
        self.init_obs(scene)
        # reset controller
        self.scene = scene
        self.state = self.ctrler.reset(scene=self.scene)
        self.start_state = deepcopy(self.state)
        self.steps = 0
        self.done = False
        self.last_action = None
        self.info = dict(
            success=False,
            scene=self.scene,
            target=self.target,
            agent_done=False,
            start_at=str(self.start_state))
        # info 中包含最短路信息
        if self.get_shortest:
            self.info.update(min_acts=self.min_actions())
        return self.get_observation()

    def reward_and_done(self):
        """判断一个应该获得的奖励，以及是否要停止，注意事件之间是相互独立，不可以叠加的"""
        done = False
        event = 'step'
        if not self.action_success:
            # if 'Move' in self.last_action:
            event = 'collision'
            # TODO if self.last_opt == 'look_up_down':
            #     event = 'camera_limit'
        if self.auto_done:
            if self.target_visible():
                event = 'success'
                done = True
                self.info['success'] = True
        elif self.last_action == 'Done':
            event = 'success' if self.target_visible() else 'fail'
            done = True
            self.info['agent_done'] = True
            self.info['success'] = (event == 'success')

        if self.steps == self.max_steps:
            done = True
            if event not in ['success', 'fail']:
                self.info['success'] = False

        return event, done

    def step(
        self,
        action: int
    ) -> Tuple:
        assert not self.done
        action = self.actions[action]
        self.action_success = True
        if action != 'Done':
            self.state, self.action_success = self.ctrler.step(action)
        self.last_action = action
        self.steps += 1
        self.info['pose'] = str(self.state)
        event, self.done = self.reward_and_done()
        reward = self.reward_dict[event]
        self.info['event'] = event
        return self.get_observation(), reward, self.done, self.info

    def add_extra_info(self, flag: bool):
        # 是否添加额外信息，例如最短路
        self.get_shortest = flag

    def min_actions(self):
        # TODO with done action
        ends = self.ctrler.visible_states(self.scene, self.target)
        acts = self.ctrler.min_actions(
            str(self.start_state), ends,
            self.actions, self.scene)
        assert acts != -1, \
            f"In {self.scene} can't reach {self.target}" +\
            f" from {self.start_state}"
        if 'Done' in self.actions:
            return acts + 1
        return acts
