import gymnasium as gym
import numpy as np
from .controller import Controller


def integrate(
    env: gym.Env,
    callbacks: list[callable] = [],
    controller: Controller = None,
    seed: int | None = None,
    episode_idx: int = 0,
):
    # Set up gym environment
    obs, info = env.reset(seed=seed)
    action = None
    # Run environment
    while True:
        # Controller
        if controller is not None:
            action = controller(env, obs, info)
        # Simulation step
        obs, reward, terminated, truncated, info = env.step(action)
        # Render
        env.render()
        # Termination criterion
        if terminated or truncated:
            break
        # Callbacks
        for callback in callbacks:
            callback(env, obs, reward, info, episode_idx=episode_idx)

    # close environment and callbacks
    for callback in callbacks:
        callback.reset()


def normalize_control(
    control: np.ndarray,
    limit: float,
):
    """Normalize control to be within the limits (Beintema et al 2020)"""
    control = np.clip(control, -limit, limit)
    control = control - np.mean(control)
    control = control / max(1, np.max(np.abs(control) / limit))
    return control


def segmentize_control(input: np.ndarray, segments: int):
    segments = np.array_split(
        input, segments
    )  # TODO how to split if segments is not a factor of array size
    return np.array([np.mean(seg) for seg in segments])
