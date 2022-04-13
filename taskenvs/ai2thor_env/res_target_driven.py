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
    embedding_path = '../vdata/word_embedding/word_embedding.hdf5'

    def __init__(
        self,
        actions: List[str],
        obs_dict: Dict[str, str],  # key: filename
        target_embedding: str,  # in ['glove' 'fasttext']
        rotate_angle: int,
        max_steps: int,
        reward_dict: Dict,
        info_scene: str,  # 用于预读数据的大小和格式以生成obs spc
        ctl_data_dir: str = '../vdata/thordata',
        obs_data_dir: Optional[str] = None
    ) -> None:
        super().__init__(
            actions, rotate_angle, max_steps, reward_dict, ctl_data_dir)
        self.t_embedding = target_embedding
        self.tgt_loader = h5py.File(self.embedding_path, "r",)
        if obs_data_dir is None:
            self.obs_data_dir = ctl_data_dir
        self.obs_dict = obs_dict
        # reading info scene to create obs info
        spath = os.path.join(self.obs_data_dir, info_scene)
        self.main_loader = {
            k: h5py.File(
                os.path.join(spath, obs_dict[k]), "r") for k in obs_dict}
        spc_dict = {
            self.t_embedding: Box(-float("inf"), float("inf"),
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
            return {self.t_embedding: ld[self.target][:]}
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
        ctl_data_dir: str = '../vdata/thordata',
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
