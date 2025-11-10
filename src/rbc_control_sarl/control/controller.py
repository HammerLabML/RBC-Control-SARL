from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym


class Policy(ABC):
    def __init__(self, env: gym.Env) -> None:
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    @abstractmethod
    def predict(self, obs) -> Any:
        pass
