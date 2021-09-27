from typing import Tuple


class AbsAgent:
    def __init__(
        self,
        model,
        env,
        *args,
        **kw_args
    ):
        pass

    def action(
        self,
        obs,
        *args,
        **kw_args
    ) -> Tuple:
        # 不应当记录梯度
        return NotImplementedError

    def sync_params(self, model):
        return NotImplementedError

    def reset_rct(self, idx):
        pass

    def close(self):
        pass
