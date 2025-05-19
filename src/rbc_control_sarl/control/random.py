from typing import Any

from .controller import Controller


class RandomController(Controller):
    def __init__(
        self,
        start: float,
        end: float,
        duration: float,
        zero: Any,
    ) -> None:
        super().__init__(start, end, duration, zero)

    def __call__(self, env, obs, info) -> float:
        if super().__call__(env, obs, info):
            self.control = env.action_space.sample()
        return self.control
