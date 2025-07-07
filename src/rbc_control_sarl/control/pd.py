import logging
from typing import List

import numpy as np
from rbc_gym.envs.rbc2D import RBCField

from .controller import Controller
from .utils import segmentize_control


class PDController(Controller):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        kp: float,
        kd: float,
        bcT: List[float],
        limit: float,
        start: float,
        end: float,
        heater_duration: float,
        heater_segments: int,
    ) -> None:
        super().__init__(start, end)
        self.kp = kp
        self.kd = kd
        self.bcT = bcT
        self.limit = limit
        self.duration = heater_duration
        self.heater_segments = heater_segments

        self._last_error = None

    def __call__(self, env, obs, info) -> float:
        if super().__call__(env, obs, info):
            # Get input and error term
            error = self.optimal_conductive_state(obs)
            error = segmentize_control(error, self.heater_segments)

            # Compute change in error
            d_error = error - (
                self._last_error if (self._last_error is not None) else error
            )

            # compute control
            control = (self.kp * error) + (self.kd * d_error / self.duration)
            self.control = np.clip(control, -1, 1)

            # Save states
            self._last_error = error

        return self.control

    def midline_temperature(self, state):
        """Singer and Bau 1917"""
        # Get input
        mid = int(state[RBCField.T].shape[0] / 2)
        input = state[RBCField.T][mid]

        # Compute error term
        T_half = 1 / 2 * (self.bcT[1] + self.bcT[0])
        T_delta = self.bcT[1] - self.bcT[0]
        error = (input - T_half) / T_delta
        return error

    def optimal_conductive_state(self, state):
        """Beintema 2020"""
        return -np.mean(state[RBCField.UY], axis=0)

    def shadow_graph(self, state):
        """Howle 1997"""
        pass
