from typing import Any

import gymnasium as gym
from .controller import Policy


class ZeroPolicy(Policy):
    # def __init__(
    #     self,
    #     env: gym.Env
    # ) -> None:
    #     super().__init__(env)

    def predict(self, env):
        return env.action_space.sample() * 0, None
