from tqdm import tqdm


class CallbackBase:
    def __init__(self, interval: int = 1):
        self.interval = interval

    def __call__(self, env, obs, reward, info, render=None, episode_idx=0) -> bool:
        return info["step"] % self.interval == 0

    def reset(self):
        pass

    def close(self):
        pass


class TqdmCallback(CallbackBase):
    def __init__(
        self,
        total: int,
        position: int = 0,
        interval: int = 1,
    ):
        super().__init__(interval=interval)
        self.pbar = tqdm(
            total=total,
            leave=False,
            position=position,
        )

    def __call__(self, env, obs, reward, info, episode_idx=0):
        if super().__call__(env, obs, reward, info):
            t = info["t"]
            self.pbar.update(t - self.pbar.n)

    def close(self):
        self.pbar.close()
