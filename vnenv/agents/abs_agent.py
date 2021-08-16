from typing import Dict, Tuple


class AbsAgent:
    def __init__(
        self,
        model,
        env,
        *args,
        **kw_args
    ):
        pass

    def model_out(self, obs, *args, **kw_args) -> Dict:
        # 需要完成数据转换
        return NotImplementedError

    def action(
        self,
        obs,
        done,  # 指示是否已经开始新的epi。
        *args,
        **kw_args
    ) -> Tuple:
        # 不应当记录梯度
        return NotImplementedError

    def sync_params(self, model):
        return NotImplementedError

    def reset_hidden(self, thread_id):
        pass

    def clear_mems(self):
        pass

    def close(self):
        pass
