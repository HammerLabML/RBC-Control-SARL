from abc import ABC, abstractmethod
from typing import Any
import gymnasium as gym
import numpy as np


class Policy(ABC):
    def __init__(self, env: gym.Env) -> None:
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def __call__(self, obs) -> Any:
        return self.call(obs)

    def predict(self, obs) -> Any:
        return np.array([self(obs[idx]) for idx in range(obs.shape[0])]), None

    @abstractmethod
    def call(self, obs) -> Any:
        pass
