import rbc_gym  # noqa: F401
import hydra
import wandb
import gymnasium as gym
from omegaconf import DictConfig

from rbc_control_sarl.callbacks import (
    TqdmCallback,
    LogNusseltNumberCallback,
    LogVisualizationCallback,
)
from rbc_control_sarl.control import PDController, integrate


@hydra.main(version_base=None, config_path="../config", config_name="pdcontrol")
def main(cfg: DictConfig) -> None:
    # Logging
    if cfg.baseline:
        tags = ["baseline"]
    else:
        tags = ["pd"]
    tags.append(f"ra{cfg.env.ra}")

    run = wandb.init(
        project="RayleighBenard-PDControl",
        dir=cfg.paths.output_dir,
        config=dict(cfg),
        tags=tags,
    )

    # Environment
    env = gym.make(
        "rbc_gym/RayleighBenardConvection2D-v0",
        rayleigh_number=cfg.env.ra,
        episode_length=cfg.env.episode_length,
        heater_duration=cfg.env.heater_duration,
        checkpoint_dir=cfg.env.checkpoint_dir,
        use_gpu=cfg.env.use_gpu,
        render_mode=cfg.env.render_mode,
    )

    # Callbacks
    callbacks = [
        TqdmCallback(total=env.unwrapped.episode_length, interval=cfg.interval),
        LogNusseltNumberCallback(interval=cfg.interval, nr_episodes=cfg.nr_episodes),
        LogVisualizationCallback(save_images=True),
    ]

    # Controller
    if not cfg.baseline:
        controller = PDController(**cfg.controller)
    else:
        controller = None

    # Rollout
    for idx in range(cfg.nr_episodes):
        integrate(
            env=env,
            controller=controller,
            callbacks=callbacks,
            seed=cfg.seed,
            episode_idx=idx,
        )

    # Finish logging
    env.close()
    for callback in callbacks:
        callback.close()
    run.finish()


if __name__ == "__main__":
    main()
