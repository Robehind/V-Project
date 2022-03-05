from collections import defaultdict
from functools import lru_cache
from ..task_env import TaskEnv
from .offline_ctrler import OfflineThorCtrler as OCer
from typing import Tuple, Any, Set
from .thordata_utils import get_scene_names, get_type
import random
from copy import deepcopy
from gym.spaces import Discrete


class BaseTargetDrivenTHOR(TaskEnv):
    # 除了观测没有定义，其他都实现好了的一个目标驱动导航环境
    def __init__(
        self,
        actions=[
            'MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight',
            'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Done'
            ],
        grid_size=0.25,
        rotate_angle=45,
        look_angle=30,
        max_steps=200,
        reward_dict={
            'step': -0.01,
            'collision': -0.1,
            'success': 5,
            'fail': 0
        }
    ) -> None:
        super().__init__()
        self.ctrler = OCer(
            grid_size=grid_size,
            rotate_angle=rotate_angle,
            look_angle=look_angle)
        self.action_space = Discrete(len(actions))
        self.actions = actions

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

    def set_tasks(self, t):
        if 'scenes' not in t:
            # 无对应关键字视为全集
            self.scenes_by_type = self.all_scenes_by_type
        else:
            self.scenes_by_type = t['scenes']
        self.scenes = get_scene_names(self.scenes_by_type)
        self.ctrler.preload_scenes(self.scenes)
        if 'targets' not in t:
            # 无对应关键字视为全集
            for s in self.scenes:
                self.targets_by_type[get_type(s)] |= self.get_objects(s)
        else:
            self.targets_by_type = {
                k: set(v) for k, v in t['targets'].items()}
        self._tasks = {
            'scenes': deepcopy(self.scenes_by_type),
            'targets': deepcopy(self.targets_by_type)}

    def _reset(
        self
    ) -> Any:
        self.scene = random.choice(self.scenes)
        tgts = self.get_objects(self.scene) & \
            self.targets_by_type[get_type(self.scene)]
        self.target = random.choice(list(tgts))
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

    def reset(self):
        self._reset()
        return self.get_observation()

    def reward_and_done(self):
        """判断一个应该获得的奖励，以及是否要停止"""
        done = False
        event = 'step'
        if not self.last_action_success:
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
        self.last_action_success = True
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
            f"can't reach {self.target} from {self.start_state}"
        if 'Done' in self.actions:
            return acts + 1
        return acts
