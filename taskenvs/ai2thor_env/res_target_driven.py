from typing import Dict, List, Optional
from .target_driven import BaseTargetDrivenTHOR as BTDTHOR
from gym.spaces import Box
from gym.spaces import Dict as DictSpc
import numpy as np
import h5py
import os
import random


class ReadFileTDenv(BTDTHOR):
    """TODO read file as observation"""

    def __init__(
        self,
        actions: List[str],
        obs_dict: Dict[str, str],  # key: filename
        target_embedding: str,  # in ['glove' 'fasttext']
        rotate_angle: int,
        max_steps: int,
        reward_dict: Dict,
        info_scene: str,  # 用于预读数据的大小和格式以生成obs spc
        ctl_data_dir: str,
        wd_path: str,
        obs_data_dir: Optional[str] = None
    ) -> None:
        super().__init__(
            actions, rotate_angle, max_steps, reward_dict, ctl_data_dir)
        self.wd_path = wd_path
        self.t_embedding = target_embedding
        self.tgt_loader = h5py.File(self.wd_path, "r",)
        if obs_data_dir is None:
            self.obs_data_dir = ctl_data_dir
        self.obs_dict = obs_dict
        # reading info scene to create obs info
        spath = os.path.join(self.obs_data_dir, info_scene)
        self.main_loader = {
            k: h5py.File(
                os.path.join(spath, obs_dict[k]), "r") for k in obs_dict}
        spc_dict = {
            'wd': Box(-float("inf"), float("inf"),
                      (300, ), dtype=np.float32)}
        for k, v in self.main_loader.items():
            sp = v[list(v.keys())[0]][:]
            if sp.dtype == np.uint8:
                spc_dict[k] = Box(0, 255,
                                  sp.shape, dtype=sp.dtype)
            else:
                spc_dict[k] = Box(-float("inf"), float("inf"),
                                  sp.shape, dtype=sp.dtype)
        self.observation_space = DictSpc(spc_dict)

    def init_obs(self, scene):
        if scene != self.scene:
            # read hdf5 file
            for k in self.main_loader:
                path = os.path.join(
                    self.obs_data_dir, scene, self.obs_dict[k])
                self.main_loader[k].close()
                self.main_loader[k] = h5py.File(path, "r")

    def get_main_obs(self):
        return {
            k: v[str(self.state)][:]
            for k, v in self.main_loader.items()}

    def get_target_obs(self):
        ld = self.tgt_loader[self.t_embedding]
        try:
            return {'wd': ld[self.target][:]}
        except KeyError:
            raise KeyError(
                f"{self.target} not supported by {self.t_embedding}")


class ZhuTDenv(BTDTHOR):
    """TODO target obs is a first-person view"""
    def __init__(
        self,
        actions: List[str],
        res_fn: str,  # filename
        rotate_angle: int,
        max_steps: int,
        reward_dict: Dict,
        info_scene: str,  # 用于预读数据的大小和格式以生成obs spc
        ctl_data_dir: str,
        obs_data_dir: Optional[str] = None
    ) -> None:
        super().__init__(
            actions, rotate_angle, max_steps, reward_dict, ctl_data_dir)
        if obs_data_dir is None:
            self.obs_data_dir = ctl_data_dir
        self.res_fn = res_fn
        # reading info scene to create obs info
        spath = os.path.join(self.obs_data_dir, info_scene)
        self.loader = h5py.File(os.path.join(spath, res_fn), "r")
        sp = self.loader[list(self.loader.keys())[0]][:]
        spc = Box(-float("inf"), float("inf"), sp.shape, dtype=sp.dtype)
        self.observation_space = DictSpc({'fc': spc, 'tgt': spc})

    def init_obs(self, scene):
        if scene != self.scene:
            # read hdf5 file
            path = os.path.join(
                self.obs_data_dir, scene, self.res_fn)
            self.loader.close()
            self.loader = h5py.File(path, "r")

    def get_main_obs(self):
        return dict(fc=self.loader[str(self.state)][:])

    def get_target_obs(self):
        states = list(self.ctrler.visible_states(self.scene, self.target))
        return dict(tgt=self.loader[random.choice(states)][:])


class MjoTDenv(BTDTHOR):
    """with bbox feature"""

    def __init__(
        self,
        actions: List[str],
        obj_path: str,
        rotate_angle: int,
        max_steps: int,
        reward_dict: Dict,
        ctl_data_dir: str,
        frame_sz: int = 224,
        obs_data_dir: Optional[str] = None
    ) -> None:
        self.fsz = frame_sz
        super().__init__(
            actions, rotate_angle, max_steps, reward_dict, ctl_data_dir)
        if obs_data_dir is None:
            self.obs_data_dir = ctl_data_dir
        self.bbox = None
        # bbox objects
        with open(obj_path) as f:
            objects = f.readlines()
            self.objects = [o.strip() for o in objects]
        # bbox obs space
        self.objs_num = len(self.objects)
        spc_dict = {
            'idx': Box(0, 2, (1, ), dtype=np.int64),
            'bbox': Box(-float("inf"), float("inf"),
                        (self.objs_num, 4), dtype=np.float32)}
        self.observation_space = DictSpc(spc_dict)

    def init_obs(self, scene):
        if scene != self.scene:
            # read hdf5 file
            path = os.path.join(
                self.obs_data_dir, scene, 'detection.hdf5')
            if self.bbox is not None:
                self.bbox.close()
            self.bbox = h5py.File(path, "r")

    def get_main_obs(self):
        objbb = self.bbox[str(self.state)]
        bbox = np.zeros((self.objs_num, 4), dtype=np.float32)
        for k in objbb:
            name = k.split("|")[0]
            if name not in self.objects:
                continue
            ind = self.objects.index(name)
            bbox[ind][0] = 1
            x1, y1, x2, y2 = objbb[k]
            bbox[ind][1] = (x1+x2)/2/self.fsz
            bbox[ind][2] = (y1+y2)/2/self.fsz
            bbox[ind][3] = abs(x2-x1) * abs(y2-y1) / self.fsz / self.fsz
        return {'bbox': bbox}

    def get_target_obs(self):
        return {'idx': np.array([self.objects.index(self.target)])}
