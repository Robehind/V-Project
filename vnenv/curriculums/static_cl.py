from .abs_cl import AbsCL
import json


class StaticCL(AbsCL):
    """Curriculum reads from json files"""
    def __init__(self, env, path) -> None:
        self.env = env
        with open(path, 'r') as f:
            self.sche = json.load(f)
        self.env.sche_update(self.sche)

    def next_sche(self, *args, **kwargs):
        pass
