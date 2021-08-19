from .base_cl import BaseCL
import json


class StaticCL(BaseCL):
    """Curriculum reads from json files"""
    def __init__(self, path) -> None:
        with open(path, 'r') as f:
            self.sche = json.load(f)

    def init_sche(self, *args, **kwargs):
        if 'sampler' in kwargs:
            kwargs['sampler'].Venv.sche_update(self.sche)
        return self.sche

    def next_sche(self, dones, sampler, *args, **kwargs):
        pass
