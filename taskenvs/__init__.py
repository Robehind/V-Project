from gym.envs.registration import register
from .task_env import TaskEnv


register(
    id='FcTdThor-v0',
    entry_point='taskenvs.ai2thor_env:ReadFileTDenv',
    kwargs=dict(
        obs_dict={'fc': 'resnet50fc.hdf5'},
        target_embedding='glove',
        info_scene='FloorPlan25')
)
register(
    id='FmTdThor-v0',
    entry_point='taskenvs.ai2thor_env:ReadFileTDenv',
    kwargs=dict(
        obs_dict={'res18fm': 'resnet18fm.hdf5'},
        target_embedding='glove',
        info_scene='FloorPlan25')
)
register(
    id='FSTdThor-v0',
    entry_point='taskenvs.ai2thor_env:ReadFileTDenv',
    kwargs=dict(
        obs_dict={'fc': 'resnet50fc.hdf5', 'score': 'resnet50score.hdf5'},
        target_embedding='glove',
        info_scene='FloorPlan25')
)
__all__ = ['TaskEnv']
