import cv2
import copy
import numpy as np
import random
import h5py
import os
import json
import importlib
from .agent_pose_state import AgentPoseState
from vnenv.utils.thordata_utils import get_type
from typing import Dict, List, Optional


class DiscreteEnvironment:
    """
    使用thordata的离散环境。
    读取数据集，模拟交互，按照dict的组织和标识来返回数据和信息。
    所有数据都是用np封装的"""
    visible_file_name = "visible_object_map.json"
    trans_file_name = 'trans.json'

    def __init__(
        self,
        offline_data_dir: str,  # 包含所有房间文件夹的路径
        action_dict: Dict[str, str],
        target_dict: Dict[str, str],
        obs_dict: Dict[str, str],
        reward_dict: Dict[str, float],
        max_steps: int,
        grid_size: float,
        rotate_angle: int,
        move_angle: int,
        horizon_angle: int,
        chosen_scenes: List[str],  # scene names random from
        chosen_targets: Optional[List[str]] = None,
        # 默认值为None时取当前房间里有的里随机一个。
        # 必须自己保证这个列表里的目标合法，例如受glove支持，在房间中可以被找到，等等
        debug: bool = False,
        seed: int = 1114,
        min_len_file: Optional[str] = None,
    ):
        random.seed(seed)

        self.min_len_file = min_len_file

        self.actions = list(action_dict.keys())
        self.action_dict = action_dict
        self.reward_dict = reward_dict
        self.offline_data_dir = offline_data_dir
        self.max_steps = max_steps

        self.debug = debug
        self.chosen_targets = chosen_targets
        self.intersect_targets = None
        self.chosen_scenes = chosen_scenes

        self.grid_size = grid_size
        self.rotate_angle = rotate_angle
        self.move_angle = move_angle
        self.horizon_angle = horizon_angle

        # 根据不同的绝对移动角度，x和z坐标的变化符合以下表格规律
        # 智能体面向z轴正方向，右手为x轴正方向时，角度为0度。向右转为正角度。
        self.move_list = [0, 1, 1, 1, 0, -1, -1, -1]
        self.move_list = [x*self.grid_size for x in self.move_list]
        # Allowed rotate angles
        self.rotations = [
            x*self.rotate_angle for x in range(0, 360//self.rotate_angle)
        ]
        # Allowed move angles
        self.move_angles = [
            x*self.move_angle for x in range(0, 360//self.move_angle)
        ]
        # Allowed horizons.
        self.horizons = [0, 30]

        # 判断环境是否需要自动为智能体停止模拟
        self.auto_done = True
        if 'Done' in self.actions:
            self.auto_done = False

        # 可变变量
        self.scene_name = None
        self.done = False
        self.reward = 0
        self.steps = 0

        # self.agent_state的数据格式沿用AgentPoseState
        self.agent_state = None
        self.start_state = None
        # self.last_agent_state = None
        self.last_action = None
        self.last_opt = None
        self.last_opt_success = True

        self.obs_dict = obs_dict
        self.obs_loader = {k: None for k in obs_dict}

        self.target_str = None
        self.target_type = list(target_dict.keys())
        self.target_dict = target_dict
        self.info = {}

        # Reading and gernerating meta data
        self.all_objects = None  # 房间支持可以找的所有物体str
        self.all_objects_id = None  # 房间支持可以找的所有物体以及其坐标，in str
        self.all_agent_states = None  # 智能体所有的可能的位姿状态，str
        self.all_visible_states = None  # 智能体在哪些位置可以看到当前目标， in str

        # 预先计算chosen_scenes中所有房间可以找到目标
        self.all_s_objects = {}  # 所有房间支持可以找的所有物体str
        self.all_s_objects_id = {}  # 所有房间支持可以找的所有物体以及其坐标，in str
        self.intersect_s_targets = {}  # 被指定且可以找到的物体，in str
        for s in self.chosen_scenes:
            s_path = os.path.join(self.offline_data_dir, s)
            with open(
                os.path.join(s_path, self.visible_file_name), "r",
            ) as f:
                visible_data = json.load(f)
            self.all_s_objects_id[s] = visible_data.keys()
            self.all_s_objects[s] = \
                set([x.split('|')[0] for x in self.all_s_objects_id[s]])
            self.all_s_objects[s] = list(self.all_s_objects[s])
            if self.chosen_targets is None:
                self.intersect_s_targets[s] = self.all_s_objects[s]
            else:
                self.intersect_s_targets[s] = list(
                    set(self.chosen_targets[get_type(s)]) &
                    set(self.all_s_objects[s])
                )
                self.intersect_s_targets[s].sort()
                if self.intersect_s_targets[s] == []:
                    raise Exception(f'In scene {s}')
        # TODO不同的目标表示可能会导致在每次重置环境时读取新的状态表示文件,未来再改善，应该写到reset里
        self.target_repre_info = {}
        self.tLoader = None
        for str_ in self.target_type:
            if str_ in ['glove', 'fasttext', 'onehot']:
                self.tLoader = h5py.File(self.target_dict[str_], "r",)
                tmp = self.tLoader[str_][list(self.tLoader[str_].keys())[0]][:]
                self.target_repre_info.update({str_: (tmp.shape, tmp.dtype)})

        # 随机读一个房间的数据，生成状态的信息，在并行化环境的时候用得上
        self.his_states = []  # in str
        self.his_len = 0
        scene_name = random.choice(self.chosen_scenes)
        self.obs_info = {}
        for type_, name_ in self.obs_dict.items():
            loader = h5py.File(
                os.path.join(offline_data_dir, scene_name, name_), "r",
            )
            tmp = loader[list(loader.keys())[0]][:]
            if '|' in type_:
                shape = (int(type_.split('|')[1]), *tmp.shape)
                self.his_len = max(self.his_len, int(type_.split('|')[1]))
            else:
                shape = tmp.shape
            self.obs_info.update({type_: (shape, tmp.dtype)})
            loader.close()

        self.data_info = {}
        self.data_info.update(self.target_repre_info)
        self.data_info.update(self.obs_info)

        # action check
        need_pitch = False
        min_rotate = 999
        for act_str in self.action_dict.values():
            if act_str is None:
                continue
            for str_ in act_str:
                angle = (int(str_[1:]) + 360) % 360
                if str_[0] == 'm':
                    assert angle in self.move_angles
                elif str_[0] == 'r':
                    if angle < min_rotate:
                        min_rotate = angle
                    assert angle % self.rotate_angle == 0
                elif str_[0] == 'p':
                    need_pitch = True
                    assert angle % self.horizon_angle == 0
                else:
                    raise Exception('Unsupported action %s' % str_)
        if not need_pitch:
            self.horizons = [0]
        if min_rotate > self.rotate_angle:
            print("Warning: min rotate angle is bigger than rotate_angle")

    def close(self):
        pass

    def init_scene(self, scene_name: Optional[str] = None) -> None:
        if scene_name is None:
            scene_name = random.choice(self.chosen_scenes)
        assert scene_name in self.chosen_scenes
        if scene_name == self.scene_name:
            return
        self.scene_name = scene_name
        self.scene_path = os.path.join(
            self.offline_data_dir, self.scene_name
        )
        with open(
            os.path.join(self.scene_path, self.trans_file_name), "r",
        ) as f:
            self.trans_data = json.load(f)
        self.all_agent_states = list(self.trans_data.keys())
        with open(
            os.path.join(self.scene_path, self.visible_file_name), "r",
        ) as f:
            self.visible_data = json.load(f)
        self.all_objects_id = self.all_s_objects_id[self.scene_name]
        self.all_objects = self.all_s_objects[self.scene_name]
        self.intersect_targets = self.intersect_s_targets[self.scene_name]
        for type_, image_ in self.obs_loader.items():
            if image_ is not None:
                self.obs_loader[type_].close()
            self.obs_loader[type_] = h5py.File(
                os.path.join(self.scene_path, self.obs_dict[type_]), "r",
            )

    def reset(
        self,
        scene_name: Optional[str] = None,
        target_str: Optional[str] = None,
        agent_state: Optional[str] = None,
        allow_no_target: bool = False,
        calc_best_len: bool = False
    ):
        """重置环境.前三个参数如果保持为None，就会随机取。allow_no_target是是否允许在没有
        设定目标的情况下运行环境。一般是不允许的。calc_best_len也是暂时保留的参数，选择是否需要
        计算最短路（很耗时，所以一般训练阶段会关掉的）
        """
        # reading metadata and obs data
        self.init_scene(scene_name)
        # set target
        if target_str is None and not allow_no_target:
            target_str = random.choice(self.intersect_targets)
        self.target_str = target_str
        self.all_visible_states = self.states_where_visible(self.target_str)

        # Initialize position
        self.set_agent_state(agent_state)
        self.start_state = copy.deepcopy(self.agent_state)
        self.his_states = [self.start_state for _ in range(self.his_len)]
        # reset variale
        self.last_action = None
        self.reward = 0
        self.done = False
        self.steps = 0
        self.info = dict(
            success=False,
            scene_name=self.scene_name,
            target=self.target_str,
            agent_done=False,
            false_action=0,
            )
        if calc_best_len:
            self.info.update(dict(best_len=self.best_path_len()[1]))

        return self.get_obs(), self.get_target_repre(self.target_str)

    def set_agent_state(self, agent_state=None):
        """设置智能体的位姿。如果agent_state为None
        """
        if agent_state is None:
            # Done也算一步，就可以出生在可终止状态
            # while 1:
            agent_state = AgentPoseState(
                pose_str=random.choice(self.all_agent_states)
            )
            agent_state.rotation = random.choice(self.rotations)
            agent_state.horizon = random.choice(self.horizons)
            # if str(agent_state) not in ban_list: break
        else:
            assert agent_state in self.all_agent_states
            # assert agent_state not in ban_list
        if isinstance(agent_state, str):
            agent_state = AgentPoseState(pose_str=agent_state)
        self.agent_state = agent_state

    def states_where_visible(self, target_str):
        """根据目标的字符串来获取当前房间所有状态中可以见到这个目标的状态，返回列表"""
        if target_str is None:
            return []
        tmp = []
        for k in self.all_objects_id:
            if k.split("|")[0] == target_str:
                tmp.extend(self.visible_data[k])
        return tmp

    def get_obs(self):
        """返回obs。可能有多个，以初始化时的关键字为索引"""
        tmp = {}
        try:
            for k, v in self.obs_loader.items():
                if '|' in k:
                    data = np.array([
                        v[str(s)][:]
                        for s in self.his_states[: int(k.split('|')[1])]
                    ])
                    tmp.update({k: data})
                else:
                    tmp.update({k: v[str(self.agent_state)][:]})
        except:
            print(self.scene_name)
        return tmp

    def rotate(self, angle: int):
        """Rotate a angle. Positive angle for turn right. Negative angle for turn left.
           Will be complished in one step."""
        self.last_opt = 'rotate'
        angle = (angle + self.agent_state.rotation + 360) % 360
        if angle in self.rotations:
            self.agent_state.rotation = angle
            self.last_opt_success = True
        else:
            self.last_opt_success = False

    def look_up_down(self, angle):
        """Look up or down by an angle. Positive angle for look down.
           Negative angle for look up. Will be complished in one step."""
        self.last_opt = 'look_up_down'
        angle = (angle + self.agent_state.horizon + 360) % 360
        if angle in self.horizons:
            self.agent_state.horizon = angle
            self.last_opt_success = True
        else:
            self.last_opt_success = False

    def move(self, angle):
        """Move towards any supported directions"""
        self.last_opt = 'move'
        abs_angle = (angle + self.agent_state.rotation + 360) % 360
        temp_rotation = self.agent_state.rotation
        self.agent_state.rotation = abs_angle
        if self.trans_data[str(self.agent_state)]:
            self.agent_state.x += self.move_list[(abs_angle//45) % 8]
            self.agent_state.z += self.move_list[(abs_angle//45+2) % 8]
            self.last_opt_success = True
        else:
            self.last_opt_success = False
        self.agent_state.rotation = temp_rotation

    def action_interpret(self, act_str):
        """翻译一个动作序列。"""
        # 空操作视为一个正常的操作
        temp_state = copy.deepcopy(self.agent_state)
        self.last_opt_success = True
        if act_str is not None:
            for str_ in act_str:
                if str_[0] == 'm':
                    self.move(int(str_[1:]))
                elif str_[0] == 'r':
                    self.rotate(int(str_[1:]))
                elif str_[0] == 'p':
                    self.look_up_down(int(str_[1:]))
                if self.last_opt_success is False:
                    self.agent_state = temp_state
                    break

    def step(self, action):
        """env的核心功能，获得一个动作，进行相关的处理，向外输出观察以及其他信息"""
        if self.done and not self.debug:
            raise Exception('Should not interact with env when env is done')
        if action not in self.actions:
            raise Exception("Unsupported action")

        self.action_interpret(self.action_dict[action])
        self.his_states = [str(self.agent_state)] + self.his_states[1:]
        # self.last_agent_state = copy.deepcopy(self.agent_state)
        self.steps += 1
        self.last_action = action
        # 分析events，给reward
        event, self.done = self.judge(action)
        self.reward = self.reward_dict[event]
        if event == 'collision':
            self.info['false_action'] += 1
        # 可以配置更多的额外环境信息在info里

        return self.get_obs(), self.reward, self.done, self.info

    def judge(self, action):
        """判断一个动作应该获得的奖励，以及是否要停止"""
        # 详细的奖惩设置都在这里
        done = False
        event = 'step'
        if not self.last_opt_success:
            if self.last_opt in ['move', 'look_up_down']:
                event = 'collision'
        if self.auto_done:
            if self.target_visiable():
                event = 'SuccessDone'
                done = True
                self.info['success'] = True
        elif action == 'Done':
            event = 'SuccessDone' if self.target_visiable() else 'FalseDone'
            done = True
            self.info['agent_done'] = True
            self.info['success'] = (event == 'SuccessDone')

        if self.steps == self.max_steps:
            done = True
            if event not in ['SuccessDone', 'FalseDone']:
                self.info['success'] = False

        return event, done

    def target_visiable(self):
        """判断目标在当前位置是否可见"""
        if str(self.agent_state) in self.all_visible_states:
            return True
        return False

    def get_target_repre(self, target_str):
        """获取目标的表示，还不够完善"""
        if target_str is None:
            return None
        repre = {}
        for str_ in self.target_type:
            if str_ in ['glove', 'fasttext', 'onehot']:
                repre[str_] = self.tLoader[str_][target_str][:]
            else:
                repre[str_] = \
                    self.get_data_of_obj(target_str, self.target_dict[str_])
        return repre

    def get_data_of_obj(
        self,
        target_str,
        file_name,
        target_json='target.json'
    ):
        '''获取一个目标的相关数据。通过检索能‘看见’这个目标的位置，来透过这个位置读取
        hdf5文件，获取这个位置的‘状态’来作为目标的表示。用file-name来指定是哪种‘状态’
        '''
        all_objID = [
            k for k in self.all_objects_id
            if k.split("|")[0] == target_str
        ]
        objID = random.choice(all_objID)
        tmp_loader = h5py.File(
            os.path.join(self.scene_path, file_name), "r",
        )
        with open(
            os.path.join(self.scene_path, target_json), "r",
        ) as f:
            state_str = json.load(f)
        data = tmp_loader[state_str[objID]][:]
        tmp_loader.close()
        return data

    def read_shortest(self):
        with open(
            os.path.join(self.scene_path, self.min_len_file), 'r'
        ) as f:
            min_len = json.load(f)
        id_ = min_len['map'][str(self.start_state)]
        all_objID = [
            k for k in self.all_objects_id
            if k.split("|")[0] == self.target_str
        ]
        min_list = [min_len[x][id_] for x in all_objID]
        # 这里直接返回理论最小值，和具体找到哪个实例不相关。
        return min(min_list)+1

    def best_path_len(self):
        """算最短路，用于计算spl.当最短路数据存在时直接读取，可以加快速度，但不再返回最佳路径的细节"""
        if self.min_len_file is not None:
            return [], self.read_shortest()
        # file loader
        nx = importlib.import_module("networkx")
        json_graph_loader = importlib.import_module("networkx.readwrite")
        with open(os.path.join(self.scene_path, 'graph.json'), 'r') as f:
            graph_json = json.load(f)
        graph = json_graph_loader.node_link_graph(graph_json).to_directed()
        start_state = self.start_state
        best_path_len = 9999
        best_path = None
        legal_states = list(graph.nodes())
        self.all_visible_states = [
            x for x in self.all_visible_states if x in legal_states
        ]

        for k in self.all_visible_states:
            try:
                path = nx.shortest_path(graph, str(start_state), k)
            except nx.exception.NetworkXNoPath:
                print(self.scene_name)
                path = nx.shortest_path(graph, str(start_state), k)
            except nx.NodeNotFound:
                print(self.scene_name)
                path = nx.shortest_path(graph, str(start_state), k)
            path_len = len(path) - 1
            # path_len = len(path)
            if path_len < best_path_len:
                best_path = path
                best_path_len = path_len

        return best_path, best_path_len

    def render(self):
        """实验性的功能"""
        pic = self.get_obs()['image'][:]
        # RGB to BGR
        pic = pic[:, :, ::-1]
        cv2.imshow("Env", pic)
        cv2.waitKey(1)
