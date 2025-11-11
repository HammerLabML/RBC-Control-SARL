import rbc_gym  # noqa: F401
import juliacall  # noqa: F401
import math
from omegaconf import DictConfig, OmegaConf
import os
from os.path import join
import numpy as np
import hydra
import gymnasium as gym
from tqdm import tqdm
import h5py
from gymnasium.wrappers import FlattenObservation, FrameStackObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from rbc_gym.wrappers import RBCNormalizeObservation
from rbc_control_sarl.control import PDPolicy, RandomPolicy, ZeroPolicy


@hydra.main(version_base=None, config_path="../config", config_name="dataset")
def main(cfg: DictConfig) -> None:
    # config convert
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # env creation
    def create_env(env_cfg):
        # env and wrappers
        env = gym.make(
            "rbc_gym/RayleighBenardConvection2D-v0",
            **env_cfg,
        )
        env = RBCNormalizeObservation(env, heater_limit=env_cfg["heater_limit"])
        env = FlattenObservation(env)
        env = FrameStackObservation(env, stack_size=1)
        return env

    env = SubprocVecEnv(
        [lambda i=i: create_env(cfg["env"]) for i in range(1, cfg["parallel"] + 1)]
    )

    # params
    shape = env.get_attr("state_shape")[0]
    steps = env.get_attr("episode_steps")[0]
    segments = env.get_attr("heater_segments")[0]

    base_seed = cfg["base_seed"]
    total_episodes = cfg["total"]
    parallel_envs = cfg["parallel"]
    control_steps = 1

    # load policy
    policy_type = cfg["type"]
    if policy_type == "ppo":
        ppo = cfg["ppo"]
        model_path = join(ppo["model_dir"], ppo["model_name"], "model")
        policy = PPO.load(model_path, env=env)
    elif policy_type == "pd":
        pd = cfg["pd"]
        policy = PDPolicy(env=env, **pd)
    elif policy_type == "random":
        random = cfg["random"]
        policy = RandomPolicy(env=env)
        control_steps = random["steps"]
    elif policy_type == "zero":
        policy = ZeroPolicy(env=env)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # Set up h5 dataset
    path = f"{cfg['out_dir']}/{cfg['type']}.h5"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as file:
        # Save commonly used parameters of the simulation
        file.attrs["episodes"] = total_episodes
        file.attrs["steps"] = steps
        file.attrs["shape"] = shape
        file.attrs["base_seed"] = base_seed
        file.attrs["ra"] = cfg["env"]["rayleigh_number"]
        file.attrs["dt"] = cfg["env"]["heater_duration"]
        file.attrs["timesteps"] = cfg["env"]["episode_length"]
        file.attrs["limit"] = cfg["env"]["heater_limit"]

        for i in range(cfg["total"]):
            # states
            file.create_dataset(
                f"states{i}",
                (steps, 3, shape[0], shape[1]),
                chunks=(1, 3, shape[0], shape[1]),
                compression="gzip",
                dtype=np.float32,
            )
            # actions
            file.create_dataset(
                f"actions{i}",
                (steps, segments),
                chunks=(steps, segments),
                compression="gzip",
                dtype=np.float32,
            )

        batches = math.ceil(total_episodes / parallel_envs)
        for base_idx in tqdm(range(batches), position=0, desc="Total Episodes"):
            # episode loop
            env.seed(base_seed + (base_idx * parallel_envs))
            obs = env.reset()
            infos = env.reset_infos
            actions = np.zeros((parallel_envs, segments))  # zero action
            for step in tqdm(range(steps), position=1, desc="Time Steps", leave=False):
                # Step environment; adapt actions every control_steps
                if step % control_steps == 0:
                    actions, _ = policy.predict(obs)

                # Save observations
                for idx in range(obs.shape[0]):
                    # don't save if id exceeds total episodes
                    id = base_idx * parallel_envs + idx
                    if id >= total_episodes:
                        continue
                    # Save state, action, and nusselt number
                    file[f"states{id}"][step] = infos[idx]["state"]
                    file[f"actions{id}"][step] = actions[idx]

                obs, _, _, infos = env.step(actions)
                env.render()

    env.close()


if __name__ == "__main__":
    main()
