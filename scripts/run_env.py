import rbc_gym  # noqa: F401
import gymnasium as gym
import hydra
from omegaconf import DictConfig

from rbc_control_sarl.control import integrate
from rbc_control_sarl.callbacks import TqdmCallback


@hydra.main(version_base=None, config_path="../config", config_name="run")
def main(cfg: DictConfig) -> None:
    env = gym.make("rbc_gym/RayleighBenardConvection2D-v0", **cfg.env)

    # Callbacks
    callbacks = [
        TqdmCallback(total=cfg.env.episode_length),
    ]

    # Rollout
    integrate(
        env=env,
        callbacks=callbacks,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    main()
