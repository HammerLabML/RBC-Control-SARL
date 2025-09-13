import numpy as np
import rbc_gym  # noqa: F401
import juliacall  # noqa: F401
from os.path import join
import hydra
from tqdm import tqdm
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from rbc_gym.wrappers import RBCNormalizeObservation
from rbc_gym.utils.visualization import update_live_control, start_live_control
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


@hydra.main(version_base=None, config_path="../config", config_name="sbsa_test")
def main(cfg: DictConfig) -> None:
    # Configure logging
    output_dir = HydraConfig.get().runtime.output_dir

    # sb3 logging
    logger = configure(join(output_dir, "log"), ["stdout", "log"])
    logger.info(f"Set log directory to {output_dir}")

    # env and wrappers
    env = gym.make(
        "rbc_gym/RayleighBenardConvection2D-v0",
        render_mode="human",
        rayleigh_number=cfg.env.ra,
        episode_length=cfg.env.episode_length,
        heater_duration=cfg.env.heater_duration,
        checkpoint=cfg.env.checkpoint,
        modes=cfg.env.control_modes,
    )
    env = RBCNormalizeObservation(env, heater_limit=cfg.env.heater_limit)
    env = FlattenObservation(env)
    env = FrameStackObservation(env, stack_size=1)

    # load trained policy
    model_path = join(cfg.experiment_dir, "model", cfg.model_name)
    policy = PPO.load(model_path, env=env)
    policy.set_logger(logger)

    #vis
    plotter = start_live_control(
        modes=env.unwrapped.modes,
        actuator_limit=env.unwrapped.actuator_limit,
        W=512,
        show_modes=True,
        max_modes=6,
    )

    # env loop
    obs, _ = env.reset()
    for _ in tqdm(range(env.unwrapped.episode_steps)):
        # get next action
        action, _ = policy.predict(obs, deterministic=True)
        update_live_control(plotter, action)

        # perform step in env
        obs, rewards, terminated, truncated, info = env.step(action)

        env.render()
        if terminated or truncated:
            break

    # close
    env.close()


if __name__ == "__main__":
    main()
