import rbc_gym  # noqa: F401
import juliacall  # noqa: F401
import logging
import os
from os.path import join

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStackObservation

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

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
    # Configure logging
    with open_dict(cfg):
        cfg.output_dir = HydraConfig.get().runtime.output_dir
        # check if already exists
        if os.path.exists(cfg.output_dir + "/wandb"):
            raise FileExistsError(f"Logging directory {cfg.output_dir} already exists")

    # wandb
    OmegaConf.resolve(cfg)
    run = wandb.init(
        project="sb3-single-agent",
        config=dict(cfg),
        sync_tensorboard=True,
        dir=cfg.output_dir,
        tags=cfg.tags,
        notes=cfg.notes,
    )
    # If we are running from slurm, append the job id to the wandb run name
    if "SLURM_JOB_ID" in os.environ:
        run.name += f"-{os.environ['SLURM_JOB_ID']}"

    # sb3 logging
    logger = configure(
        join(cfg.output_dir, "log"), ["stdout", "log", "json", "tensorboard"]
    )
    logger.info(f"Set log directory to {cfg.output_dir}")
    logger.info(f"Logging results wandb run {run.project}/{run.name}")

    # Construct the evaluation and training environments
    def create_env(env_cfg, env_id="0", render_mode=None):
        env = gym.make(
            "rbc_gym/RayleighBenardConvection2D-v0",
            render_mode=render_mode,
            rayleigh_number=env_cfg.ra,
            episode_length=env_cfg.episode_length,
            heater_duration=env_cfg.heater_duration,
            checkpoint_dir=env_cfg.checkpoint_dir,
            use_gpu=env_cfg.use_gpu,
        )
        env = FlattenObservation(env)
        env = FrameStackObservation(env, cfg.sb3.frame_stack)
        return env

    train_env = SubprocVecEnv(
        [
            lambda i=i: create_env(cfg.train_env, f"train_{i}")
            for i in range(1, cfg.sb3.nr_processes + 1)
        ]
    )
    val_env = SubprocVecEnv(
        [
            lambda i=i: create_env(cfg.val_env, f"val_{i}")
            for i in range(1, cfg.sb3.nr_eval_processes + 1)
        ]
    )

    # Parameters
    steps_per_iteration = cfg.sb3.ppo.episodes_update * int(
        cfg.train_env.episode_length / cfg.train_env.action_duration
    )

    # Policy model
    nr_neurons = cfg.sb3.ppo.nr_neurons
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[nr_neurons, nr_neurons], vf=[nr_neurons, nr_neurons]),
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        n_steps=steps_per_iteration,
        learning_rate=cfg.sb3.ppo.lr,
        batch_size=cfg.sb3.ppo.batch_size,
        gamma=cfg.sb3.ppo.gamma,
        ent_coef=cfg.sb3.ppo.ent_coef,
        verbose=1,
    )

    # Callbacks
    dir_model = join(cfg.output_dir, "model")
    dir_log = join(cfg.output_dir, "log")
    # train checkpoint

    os.makedirs(dir_model, exist_ok=True)
    checkpoint_cb_training = CheckpointCallback(
        save_freq=cfg.sb3.train_checkpoint_every
        * int(cfg.train_env.episode_length / cfg.train_env.action_duration),
        save_path=dir_model,
        name_prefix="PPO_train",
    )

    # evaluation callback
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=dir_model,
        log_path=dir_log,
        eval_freq=cfg.sb3.eval_every * steps_per_iteration,
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
        total_timesteps=cfg.sb3.train_steps, progress_bar=True, callback=callbacks
    )

    train_env.close()
    val_env.close()
    run.finish()


if __name__ == "__main__":
    main()
