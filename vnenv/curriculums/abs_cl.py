class AbsCL:
    def __init__(self, env, *args, **kwargs) -> None:
        pass

    def next_task(self, update_steps, *args, **kwargs):
        return False
