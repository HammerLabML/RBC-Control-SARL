import math
from omegaconf import DictConfig, OmegaConf
import rbc_gym  # noqa: F401
import os
from os.path import join
import numpy as np
import hydra
import gymnasium as gym
from tqdm import tqdm
import h5py
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from rbc_gym.wrappers import RBCNormalizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


@hydra.main(version_base=None, config_path="../config", config_name="dataset")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    
    def create_env(env_cfg):
        # env and wrappers
        env = gym.make(
            "rbc_gym/RayleighBenardConvection2D-v0",
            **env_cfg,
        )
        env = RBCNormalizeObservation(env, heater_limit=env_cfg.heater_limit)
        env = FlattenObservation(env)
        env = FrameStackObservation(env, stack_size=1)
        return env

    env = SubprocVecEnv(
        [
            lambda i=i: create_env(cfg.env)
            for i in range(1, cfg.parallel + 1)
        ]
    )

    # params
    shape = env.get_attr("state_shape")[0]
    steps = env.get_attr("episode_steps")[0]
    segments = env.get_attr("heater_segments")[0]

    base_seed = cfg.base_seed
    total_epsiodes = cfg.dataset.total
    parallel_envs = cfg.parallel
    control_steps = cfg.control_steps

    # load trained policy
    model_path = join(cfg.experiment_dir, "model", cfg.model_name)
    policy = PPO.load(model_path, env=env)

    # Set up h5 dataset
    dir = "data/datasets/2D-control"
    path = f"{dir}/ra{cfg.ra}/{cfg.dataset.type}.h5"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as file:
        # Save commonly used parameters of the simulation
        file.attrs["ra"] = cfg.ra

        for i in range(cfg.dataset.total):
            # states
            file.create_dataset(
                f"s-{i}",
                (steps, 3, shape[0], shape[1]),
                chunks=(1, 3, shape[0], shape[1]),
                compression="gzip",
                dtype=np.float32,
            )
            # actions
            file.create_dataset(
                f"a-{i}",
                (steps, segments),
                chunks=(steps, segments),
                compression="gzip",
                dtype=np.float32,
            )

    # Run environment and save observations
    def get_actions(obs):
        t = cfg.dataset.type
        if t == "ppo":
            return policy.predict(obs)[0]
        elif t == "random":
            return np.array([env.action_space.sample() for _ in range(obs.shape[0])])
        elif t == "zero":
            return np.zeros((parallel_envs, segments))
        else:
            raise ValueError(f"Unknown dataset type: {cfg.dataset.type}")
        

    batches = math.ceil(total_epsiodes / parallel_envs)
    for base_idx in tqdm(range(batches), position=0, desc="Total Episodes"):
        # episode loop
        env.seed(base_seed + (base_idx * parallel_envs))
        obs = env.reset()
        infos = env.reset_infos
        actions = np.zeros((parallel_envs, segments)) #zero action
        for step in tqdm(range(steps), position=1, desc="Time Steps", leave=False):
            # Save observations
            for idx in range(obs.shape[0]):
                # don't save if id exceeds total episodes
                id = base_idx * parallel_envs + idx
                if id >= total_epsiodes:
                    continue
                # Save state, action, and nusselt number
                with h5py.File(path, "r+") as file:
                    file[f"s-{id}"][step] = infos[idx]["state"]
                    file[f"a-{id}"][step] = actions[idx]
            
            # Step environment; adapt actions every control_steps
            if step % control_steps == 0:
                actions = get_actions(obs)

            obs, _, dones, _ = env.step(actions)
            if dones.any():
                break

    env.close()


if __name__ == "__main__":
    main()
