from abc import ABC


class Controller(ABC):
    def __init__(
        self,
        start: float,
        end: float,
    ) -> None:
        # Params
        self.start = start
        self.end = end
        self.control = None

    def __call__(self, env, obs, info) -> bool:
        # check if the controller should apply a new action
        if info["t"] < self.start:
            return False
        elif info["t"] > self.end:
            self.control = None
            return False
        return True
