from typing import Dict, List
from .target_driven import BaseTargetDrivenTHOR as BTDTHOR
from gym.spaces import Box
from gym.spaces import Dict as DictSpc
import numpy as np
import h5py
import os


# 一些物体可能环境有，但是它的单词并不受glove支持
class FCTargetDrivenEnv(BTDTHOR):
    """fc: resnet50 target: glove"""
    resnetFC_path = '../vdata/thordata'
    resnetFC_name = 'resnet50fc_no_norm.hdf5'
    glove_path = '../vdata/word_embedding/word_embedding.hdf5'

    def __init__(
        self,
        actions: List[str],
        reward_dict: Dict,
        grid_size=0.25,
        rotate_angle=45,
        look_angle=30,
        max_steps=200,
    ) -> None:
        super().__init__(
            actions, grid_size, rotate_angle,
            look_angle, max_steps, reward_dict)
        self.observation_space = DictSpc({
            'fc': Box(-float("inf"), float("inf"),
                      (2048, ), dtype=np.float32),
            'glove': Box(-float("inf"), float("inf"),
                         (300, ), dtype=np.float32)})
        self.tgt_loader = h5py.File(self.glove_path, "r",)
        self.main_loader = {'fc': None}

    def _reset(self):
        super()._reset()
        # read hdf5 file
        path = os.path.join(
            self.resnetFC_path, self.scene, self.resnetFC_name)
        if self.main_loader['fc'] is not None:
            self.main_loader['fc'].close()
        self.main_loader = {'fc': h5py.File(path, "r")}

    def get_main_obs(self):
        return {
            k: v[str(self.state)][:]
            for k, v in self.main_loader.items()}

    def get_target_obs(self):
        ld = self.tgt_loader['glove']
        try:
            return {'glove': ld[self.target][:]}
        except KeyError:
            raise KeyError(f"{self.target} not supported by glove")
