import rbc_gym  # noqa: F401
import juliacall  # noqa: F401
import hydra
import wandb
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from rbc_control_sarl.callbacks import (
    TqdmCallback,
    LogNusseltNumberCallback,
    LogVisualizationCallback,
)
from rbc_control_sarl.control import PDPolicy, integrate


@hydra.main(version_base=None, config_path="../config", config_name="pd")
def main(cfg: DictConfig) -> None:
    # config convert
    cfg = OmegaConf.to_container(cfg, resolve=True)
    output_dir = cfg["paths"]["output_dir"]

    # Logging
    if cfg["baseline"]:
        tags = ["baseline"]
    else:
        tags = ["pd"]
    tags.append(f"ra{cfg['env']['rayleigh_number']}")

    run = wandb.init(
        project="RayleighBenard-PDControl",
        dir=output_dir,
        config=cfg,
        tags=tags,
    )

    # Environment
    env = gym.make("rbc_gym/RayleighBenardConvection2D-v0", **cfg["env"])

    # Callbacks
    interval = cfg["interval"]
    nr = cfg["nr_episodes"]
    callbacks = [
        TqdmCallback(total=env.unwrapped.episode_length, interval=interval),
        LogNusseltNumberCallback(interval=interval, nr_episodes=nr),
        LogVisualizationCallback(save_images=True),
    ]

    # Controller
    if not cfg["baseline"]:
        controller = PDPolicy(**cfg["pd"], env=env)
    else:
        controller = None

    # Rollout
    for idx in range(nr):
        integrate(
            env=env,
            policy=controller,
            callbacks=callbacks,
            seed=cfg["seed"],
            episode_idx=idx,
        )

    # Finish logging
    env.close()
    for callback in callbacks:
        callback.close()
    run.finish()


if __name__ == "__main__":
    main()
