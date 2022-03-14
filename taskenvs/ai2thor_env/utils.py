from tqdm import tqdm
import h5py
import os
from typing import Dict, Optional
DEFAULT_Y = 0.91  # THOR环境是平坦的，因此智能体的高为一个定植
# 默认为z轴为旋转参考轴，y轴为高度轴


class AgentPoseState:
    """表示智能体在离散THOR环境中的位置姿态的类"""

    def __init__(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        rotation: float = 0,
        horizon: float = 0,
        pose_str: Optional[str] = None
    ) -> None:
        if pose_str is not None:
            x, z, rotation, horizon = [float(x) for x in pose_str.split("|")]
            if y is None:
                y = DEFAULT_Y
        self.x = round(x, 2)
        self.y = y
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)

    def __eq__(self, other) -> bool:
        """比较两个位姿是否相同"""
        if isinstance(other, AgentPoseState):
            return (
                self.x == other.x
                and
                # thor中y值一定相同
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )

    def __str__(self) -> str:
        """返回字符串形式的智能体位姿状态, x与z保留两位小数
        """
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self) -> Dict[str, float]:
        """只返回坐标"""
        return dict(x=self.x, y=self.y, z=self.z)


def get_scene_names(scenes):
    """根据参数生成完整的房间的名字"""
    tmp = []
    for k in scenes.keys():
        ranges = [x for x in scenes[k].split(',')]
        number = []
        for a in ranges:
            ss = [int(x) for x in a.split('-')]
            number += range(ss[0], ss[-1]+1)
        number = list(set(number))
        tmp += [make_scene_name(k, i) for i in number]
    return tmp


def get_type(scene_name):
    """根据房间名称返回该房间属于哪个类型"""
    mapping = {'2': 'living_room', '3': 'bedroom', '4': 'bathroom'}
    num = scene_name.split('_')[0].split('n')[-1]
    if len(num) < 3:
        return 'kitchen'
    return mapping[num[0]]


def make_scene_name(scene_type, num):
    """根据房间的类别和序号生成房间的名称
    例如，scene_type = kitchen的第num = 5个房间，为FloorPlan5
    """
    mapping = {"kitchen": '', "living_room": '2',
               "bedroom": '3', "bathroom": '4'}
    front = mapping[scene_type]
    if num >= 10 or front == '':
        return "FloorPlan" + front + str(num)
    return "FloorPlan" + front + "0" + str(num)


def states_num(scenes, datadir, preload):

    scene_names = get_scene_names(scenes)
    count = 0
    pbar = tqdm(total=len(scene_names), desc='Gathering...', leave=False)
    for s in scene_names:
        RGBloader = h5py.File(os.path.join(datadir, s, preload), "r",)
        num = len(list(RGBloader.keys()))
        count += num
        pbar.update(1)
        RGBloader.close()
    return count
