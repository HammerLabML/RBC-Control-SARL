import rbc_gym  # noqa: F401
import juliacall  # noqa: F401
import os
from os.path import join
import hydra
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import wandb
import yaml
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from rbc_gym.wrappers import RBCNormalizeObservation, RBCRewardShaping
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb import Table, Video


@hydra.main(version_base=None, config_path="../config", config_name="sbsa_eval")
def main(cfg: DictConfig) -> None:
    # Configure logging
    output_dir = HydraConfig.get().runtime.output_dir

    # sb3 logging
    logger = configure(
        join(output_dir, "log"), ["stdout", "log", "json", "tensorboard"]
    )
    logger.info(f"Set log directory to {output_dir}")

    model_name = cfg.model_name
    # get train config
    with open(join(cfg.experiment_dir, ".hydra/config.yaml")) as file:
        config = DictConfig(yaml.safe_load(file))
        model_path = join(cfg.experiment_dir, "model", model_name)
    # resolve config and put in dict
    with open_dict(cfg):
        OmegaConf.resolve(config)
        cfg.train_cfg = config
    OmegaConf.resolve(cfg)
    logger.info(f"Loaded config from {cfg.experiment_dir}/.hydra/config.yaml")

    # wandb
    run = wandb.init(
        project="sb3-single-agent",
        config=dict(cfg),
        sync_tensorboard=True,
        dir=output_dir,
        tags=cfg.tags,
    )

    # If we are running from slurm, append the job id to the wandb run name
    if "SLURM_JOB_ID" in os.environ:
        run.name += f"-{os.environ['SLURM_JOB_ID']}"

    # Get env, wrappers and policy
    env = make_vec_env(
        lambda: FrameStackObservation(
            FlattenObservation(
                RBCRewardShaping(
                    RBCNormalizeObservation(
                        gym.make(
                            "rbc_gym/RayleighBenardConvection2D-v0",
                            render_mode=cfg.env.render_mode,
                            rayleigh_number=cfg.env.ra,
                            episode_length=cfg.env.episode_length,
                            heater_duration=cfg.env.heater_duration,
                            checkpoint=cfg.env.checkpoint,
                            use_gpu=cfg.env.use_gpu,
                        ),
                        heater_limit=cfg.env.heater_limit,
                    ),
                    shaping_weight=0,
                )
            ),
            stack_size=config.sb3.frame_stack,
        ),
        vec_env_cls=DummyVecEnv,
    )
    policy = PPO.load(model_path, env=env)
    policy.set_logger(logger)

    # Enjoy trained agent
    nusselts = []
    cell_dists = []
    times = []
    episode = []
    for idx in range(cfg.nr_episodes):
        logger.info(f"Evaluating model on episode {idx}")
        # data holders
        screens = []
        actions = []
        # logging
        wandb.define_metric(f"ep{idx}/time")
        wandb.define_metric(
            f"ep{idx}/nusselt_state", step_metric=f"ep{idx}/time", summary="mean"
        )
        wandb.define_metric(
            f"ep{idx}/nusselt_obs", step_metric=f"ep{idx}/time", summary="mean"
        )
        wandb.define_metric(
            f"ep{idx}/reward", step_metric=f"ep{idx}/time", summary="mean"
        )
        wandb.define_metric(
            f"ep{idx}/cell_dist", step_metric=f"ep{idx}/time", summary="mean"
        )
        # reset env
        obs = env.reset()
        dones = np.zeros(1)
        while not dones.any():
            # get next action
            action, _ = policy.predict(obs, deterministic=True)
            # save data
            if cfg.env.render_mode == "human":
                env.render()
            elif cfg.env.render_mode == "rgb_array":
                screens.append(env.render().transpose(2, 0, 1))
            actions.append(action.squeeze())
            # perform step in env
            obs, rewards, dones, info = env.step(action)

            # log data
            wandb.log(
                {
                    f"ep{idx}/time": info[0]["t"],
                    f"ep{idx}/nusselt_state": info[0]["nusselt_state"],
                    f"ep{idx}/nusselt_obs": info[0]["nusselt_obs"],
                    f"ep{idx}/reward": rewards[0],
                    f"ep{idx}/cell_dist": info[0]["cell_dist"],
                }
            )
            nusselts.append(info[0]["nusselt_obs"])
            cell_dists.append(info[0]["cell_dist"])
            times.append(info[0]["t"])
            episode.append(idx)
        # plot data
        plot_actions(actions, output_dir, idx, fps=cfg.fps)
        wandb.log(
            {
                f"ep{idx}/video": Video(np.asarray(screens), fps=cfg.fps, format="mp4"),
            }
        )

    # log overall mean nusselt
    nusselt_mean = np.mean(nusselts)
    logger.info(f"Mean nusselt number: {nusselt_mean}")
    df = pd.DataFrame(
        {
            "nusselt": np.array(nusselts),
            "time": np.array(times),
            "episode": np.array(episode),
        }
    )
    wandb.log({"nusselt_table": Table(dataframe=df)})
    wandb.log({"mean_nusselt": nusselt_mean})

    # log overall mean cell_dists
    cell_dist_mean = np.mean(cell_dists)
    logger.info(f"Mean cell dist : {cell_dist_mean}")
    df = pd.DataFrame(
        {
            "cell_dist": np.array(cell_dists),
            "time": np.array(times),
            "episode": np.array(episode),
        }
    )
    wandb.log({"cell_dist_table": Table(dataframe=df)})
    wandb.log({"mean_cell_dist": cell_dist_mean})

    # close
    env.close()
    run.finish()


def plot_actions(actions, out_dir, episode_idx, fps=2):
    # Plot nusselt number
    fig, ax = plt.subplots()
    # Plot amplitude
    ax.set_xlabel("segements")
    ax.set_ylabel("amplitude")
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis="y")
    ax.grid()
    # plot actions
    artists = []
    for action in actions:
        container = ax.plot(range(len(action)), action, color="blue")
        artists.append(container)

    ani = animation.ArtistAnimation(fig=fig, artists=artists)
    writer = animation.FFMpegWriter(fps=fps)
    path = f"{out_dir}/actions_ep{episode_idx}.mp4"
    ani.save(path, writer=writer)
    wandb.log(
        {
            f"ep{episode_idx}/actions": Video(path, format="mp4"),
        }
    )


if __name__ == "__main__":
    main()
