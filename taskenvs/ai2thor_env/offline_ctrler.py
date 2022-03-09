from functools import lru_cache
import os
import json
from typing import List, Optional, Set
import copy
import random
from taskenvs.ai2thor_env.agent_pose_state import AgentPoseState


class OfflineThorCtrler:
    visible_file = "visible_map.json"
    trans_file = 'trans.json'
    room_type = ['kitchen', 'bedroom', 'living_room', 'bathroom']

    def __init__(
        self,
        grid_size=0.25,
        rotate_angle=45,
        look_angle=30,
        horizon_limits=[0, 30],
        data_dir='../vdata/thordata'
    ):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.rotate_angle = rotate_angle
        self.look_angle = look_angle
        # 向上向下看的角度可能是有限制的
        self.horizon_limits = horizon_limits
        self.move_list = [0, 1, 1, 1, 0, -1, -1, -1]
        self.move_list = [x*self.grid_size for x in self.move_list]
        # 根据不同的绝对移动角度，x和z坐标的变化符合以下表格规律
        # 智能体面向z轴正方向，右手为x轴正方向时，角度为0度。向右转为正角度。
        # 向上看为负角度，向下看为正角度
        self.actions = {
            'MoveAhead': self.move(0), 'MoveBack': self.move(180),
            'MoveLeft': self.move(-90), 'MoveRight': self.move(90),
            'RotateRight': self.rotate(self.rotate_angle),
            'RotateLeft': self.rotate(-self.rotate_angle),
            'LookUp': self.look(-self.look_angle),
            'LookDown': self.look(self.look_angle)
        }

        self.scene = None
        self.state = None  # in AgentPoseState
        self.start_state = None
        self.last_action = None
        self.last_action_success = True

        self.metadata = {}
        self.all_states = None  # 智能体所有的可能的位姿状态，str
        self.trans_data = {}
        self.visible_data = {}

    def preload_scenes(self, scenes: List[str]):
        # pre load scene data to speed up
        for s in scenes:
            if s not in self.visible_data:
                self.load_scene(s)

    @lru_cache(None)
    def visible_states(self, scene: str, obj: str) -> Set:
        if scene not in self.visible_data:
            self.load_scene(scene)
        tmp = set()
        for k in self.visible_data[scene]:
            if k.split("|")[0] == obj:
                tmp |= set(self.visible_data[scene][k])
        return tmp

    def load_scene(self, scene: str):
        # load a single scene's data
        scene_path = os.path.join(
            self.data_dir, scene
        )
        with open(
            os.path.join(scene_path, self.trans_file), "r",
        ) as f:
            self.trans_data[scene] = json.load(f)
        with open(
            os.path.join(scene_path, self.visible_file), "r",
        ) as f:
            self.visible_data[scene] = json.load(f)

    def objectIDs(self, scene) -> Set:
        if scene not in self.visible_data:
            self.load_scene(scene)
        return set(self.visible_data[scene].keys())

    def reset(
        self,
        scene: str,
        state: Optional[str] = None
    ):
        self.scene = scene
        self.all_states = list(self.trans_data[scene].keys())

        # Initialize position
        if state is None:
            self.state = random.choice(self.all_states)
        else:
            assert state in self.all_states
            self.state = state
        self.state = AgentPoseState(pose_str=self.state)
        # fix horizon if need
        if self.look_angle == 0:
            self.state.horizon = 0
        # if 90, don't init rotation to 45*(2n-1)
        if self.rotate_angle == 90:
            self.state.rotation = random.choice([0, 90, 180, 270])
        self.start_state = copy.deepcopy(self.state)
        # reset variale
        self.last_action = None
        self.metadata = dict(
            scene=self.scene,
            start_at=str(self.start_state)
            )
        return self.state

    def rotate(self, angle):
        def _rotate(state, scene):
            assert angle != 0
            state.rotation = (angle + state.rotation + 360) % 360
            return state, True
        return _rotate

    def look(self, angle):
        def _look(state, scene):
            assert angle != 0
            _angle = (angle + state.horizon + 360) % 360
            if _angle in self.horizon_limits:
                state.horizon = _angle
                action_success = True
            else:
                action_success = False
            return state, action_success
        return _look

    def move(self, angle):
        def _move(state, scene):
            abs_angle = (angle + state.rotation + 360) % 360
            temp_rotation = state.rotation
            state.rotation = abs_angle
            if self.trans_data[scene][str(state)]:
                state.x += self.move_list[(abs_angle//45) % 8]
                state.z += self.move_list[(abs_angle//45+2) % 8]
                action_success = True
            else:
                action_success = False
            state.rotation = temp_rotation
            return state, action_success
        return _move

    def step(self, action: str):
        self.state, self.last_action_success = \
            self.actions[action](self.state, self.scene)
        self.last_action = action
        self.metadata['pose'] = str(self.state)
        return self.state, self.last_action_success

    def min_actions(
        self,
        s: str,
        g: List[str],
        actions: List[str],
        scene: str
    ):
        """算从s到g的最少动作数量"""
        if scene not in self.visible_data:
            self.load_scene(scene)
        from collections import defaultdict
        if s in g:
            return 0
        s = AgentPoseState(pose_str=s)
        c_nodes = []
        n_nodes = [s]
        mark = defaultdict(int)
        mark[str(s)] = 1
        length = 0
        while n_nodes:
            c_nodes = n_nodes
            n_nodes = []
            length += 1
            for nn in c_nodes:
                for act in actions:
                    if act == 'Done':
                        continue
                    _nn = copy.deepcopy(nn)
                    new_n, flag = self.actions[act](_nn, scene)
                    if not flag:
                        continue
                    if str(new_n) in g:
                        return length
                    if mark[str(new_n)] == 0:
                        n_nodes.append(new_n)
                        mark[str(new_n)] = 1
        return -1


if __name__ == "__main__":
    import h5py
    import cv2
    path = input("input path:")
    ctrler = OfflineThorCtrler(data_dir=path)
    scene = input("scene:")
    ctrler.preload_scenes([scene])
    pose = ctrler.reset(scene=scene)
    cmd_map = {
        119: 'MoveAhead',
        100: 'RotateRight',
        97: 'RotateLeft',
        107: 'LookDown',
        105: 'LookUp'}
    # frames hdf5
    s_path = os.path.join(path, scene, 'frame.hdf5')
    frames = h5py.File(s_path, 'r')
    print(pose)
    cv2.imshow('vis', frames[str(pose)][:][:, :, ::-1])
    key = cv2.waitKey(0)
    while key != 27:
        if key in cmd_map:
            pose, suc = ctrler.step(cmd_map[key])
            print(pose, suc)
            cv2.imshow('vis', frames[str(pose)][:][:, :, ::-1])
            key = cv2.waitKey(0)
        else:
            print("Pressing ", key)
