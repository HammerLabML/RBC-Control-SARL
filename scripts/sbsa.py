import rbc_gym  # noqa: F401
import juliacall  # noqa: F401
import logging
import os
from os.path import join

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from rbc_gym.wrappers import (
    RBCNormalizeObservation,
    RBCNormalizeReward,
    RBCRewardShaping,
)

import hydra
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback

from rbc_control_sarl.callbacks import NusseltCallbackSB3
import torch


logger = logging.getLogger("sb3")


@hydra.main(version_base=None, config_path="../config", config_name="sbsa")
def main(cfg: DictConfig) -> None:
    # config convert
    cfg = OmegaConf.to_container(cfg, resolve=True)
    output_dir = cfg["paths"]["output_dir"]

    # check if out dir already exists
    if os.path.exists(output_dir + "/wandb"):
        raise FileExistsError(f"Logging directory {output_dir} already exists")

    # wandb
    run = wandb.init(
        project="sb3-single-agent",
        config=cfg,
        sync_tensorboard=True,
        dir=output_dir,
        tags=cfg["tags"],
        notes=cfg["notes"],
    )
    # If we are running from slurm, append the job id to the wandb run name
    if "SLURM_JOB_ID" in os.environ:
        run.name += f"-{os.environ['SLURM_JOB_ID']}"

    # sb3 logging
    logger = configure(
        join(output_dir, "log"), ["stdout", "log", "json", "tensorboard"]
    )
    logger.info(f"Set log directory to {output_dir}")
    logger.info(f"Logging results wandb run {run.project}/{run.name}")

    # Construct the evaluation and training environments
    def create_env(env_cfg):
        env = gym.make(
            "rbc_gym/RayleighBenardConvection2D-v0",
            **env_cfg,
        )
        env = RBCNormalizeObservation(env, heater_limit=env_cfg["heater_limit"])
        env = RBCNormalizeReward(env, ra=env_cfg["rayleigh_number"], s=0.1, a=0.4)
        env = RBCRewardShaping(env, shaping_weight=cfg["reward_shaping"])
        env = FlattenObservation(env)
        env = FrameStackObservation(env, cfg["sb3"]["frame_stack"])
        return env

    train_env = SubprocVecEnv(
        [
            lambda i=i: create_env(cfg["train_env"])
            for i in range(1, cfg["sb3"]["nr_processes"] + 1)
        ]
    )
    val_env = SubprocVecEnv(
        [
            lambda i=i: create_env(cfg["val_env"])
            for i in range(1, cfg["sb3"]["nr_processes"] + 1)
        ]
    )

    # Parameters
    sb3_cfg = cfg["sb3"]
    ppo_cfg = cfg["sb3"]["ppo"]
    steps_per_iteration = ppo_cfg["episodes_update"] * int(
        cfg["train_env"]["episode_length"] / cfg["train_env"]["heater_duration"]
    )

    # Policy model
    nr_neurons = ppo_cfg["nr_neurons"]
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[nr_neurons, nr_neurons], vf=[nr_neurons, nr_neurons]),
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        n_steps=steps_per_iteration,
        learning_rate=ppo_cfg["lr"],
        batch_size=ppo_cfg["batch_size"],
        gamma=ppo_cfg["gamma"],
        ent_coef=ppo_cfg["ent_coef"],
        verbose=1,
    )

    # Callbacks
    dir_model = join(output_dir, "model")
    dir_log = join(output_dir, "log")
    # train checkpoint

    os.makedirs(dir_model, exist_ok=True)
    checkpoint_cb_training = CheckpointCallback(
        save_freq=sb3_cfg["train_checkpoint_every"]
        * int(cfg["train_env"]["episode_length"] / cfg["train_env"]["heater_duration"]),
        save_path=dir_model,
        name_prefix="PPO_train",
    )

    # evaluation callback
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=dir_model,
        log_path=dir_log,
        eval_freq=sb3_cfg["eval_every"] * steps_per_iteration,
        deterministic=True,
        render=False,
    )

    callbacks = [
        NusseltCallbackSB3(),
        eval_cb,
        checkpoint_cb_training,
        WandbCallback(
            verbose=1,
        ),
    ]

    # Train the model
    model.set_logger(logger)
    model.learn(
        total_timesteps=sb3_cfg["train_steps"], progress_bar=True, callback=callbacks
    )

    train_env.close()
    val_env.close()
    run.finish()


if __name__ == "__main__":
    main()
