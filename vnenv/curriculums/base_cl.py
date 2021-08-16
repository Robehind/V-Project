class BaseCL:
    def __init__(self) -> None:
        self.sche_count = 0

    def sche_create(self):
        return list()

    def init_sche(self, sampler):
        pass

    def next_sche(self, dones, sampler, *args, **kwargs):
        pass
        # sche = self.sche_create()
        # sampler.Venv.sche_update(sche)
