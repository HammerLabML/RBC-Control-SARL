import logging
import os
import pathlib
import tempfile
from typing import Optional

import numpy as np
import seaborn as sns
import sympy as sp
import wandb
from matplotlib import animation
from matplotlib import pyplot as plt
from PIL import Image

from rbc_control_sarl.callbacks import CallbackBase


class LogNusseltNumberCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
        nr_episodes: int = 1,
    ):
        super().__init__(interval=interval)
        for idx in range(nr_episodes):
            wandb.define_metric(f"ep{idx}/time")
            wandb.define_metric(
                f"ep{idx}/nusselt_state", step_metric=f"ep{idx}/time", summary="mean"
            )
            wandb.define_metric(
                f"ep{idx}/nusselt_obs", step_metric=f"ep{idx}/time", summary="mean"
            )

    def __call__(self, env, obs, reward, info, episode_idx):
        if super().__call__(env, obs, reward, info):
            wandb.log(
                {
                    f"ep{episode_idx}/time": info["t"],
                    f"ep{episode_idx}/nusselt_state": info["nusselt_state"],
                    f"ep{episode_idx}/nusselt_obs": info["nusselt_obs"],
                }
            )


class LogVisualizationCallback(CallbackBase):
    def __init__(
        self,
        fps: int = 4,
        interval: Optional[int] = 1,
        save_video: bool = True,
        save_images: bool = False,
    ):
        super().__init__(interval=interval)
        self.fps = fps
        self.screens = []
        self.save_video = save_video
        self.save_images = save_images

    def __call__(self, env, obs, reward, info, episode_idx=0):
        if super().__call__(env, obs, reward, info):
            screen = env.render()
            self.screens.append(screen.transpose(2, 0, 1))
            self.ep_idx = episode_idx

            if self.save_images:
                im = wandb.Image(screen, caption="state")
                os.makedirs("states/", exist_ok=True)
                Image.fromarray(screen).save(f"states/state_{info['t']}.png")
                wandb.log(
                    {
                        f"ep{episode_idx}/state": im,
                    }
                )

    def reset(self):
        if self.save_video:
            wandb.log(
                {
                    f"ep{self.ep_idx}/state_video": wandb.Video(
                        np.asarray(self.screens), fps=self.fps, format="mp4"
                    )
                }
            )
        self.screens = []


class LogActionCallback(CallbackBase):
    def __init__(
        self,
        interval: Optional[int] = 1,
        save_video: bool = True,
        save_images: bool = False,
    ):
        super().__init__(interval=interval)
        self.actions = []
        self.save_video = save_video
        self.save_images = save_images

        # suppress matplotlib logging
        logger = logging.getLogger("matplotlib.animation")
        logger.setLevel(logging.ERROR)

    def __call__(self, env, obs, reward, info, episode_idx=0):
        if super().__call__(env, obs, reward, info):
            # get and save action
            self.ep_idx = episode_idx
            action = env.action_effective

            # plot setup
            fig = plt.figure(figsize=(5, 3))
            sns.set_theme()

            plt.xlabel("Spatial x")
            plt.xlim(0, 2 * np.pi)
            plt.xticks(ticks=[0, np.pi, 2 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])

            plt.ylabel(r"Control Input $\hat{T}$")
            plt.ylim(-1, 1)
            plt.yticks(ticks=[-0.75, 0, 0.75], labels=["-C", "0", "C"])

            # plot and save to wandb
            y = sp.Symbol("y")
            action_lambda = sp.lambdify(y, action, "numpy")
            x_vals = np.linspace(0, 2 * sp.pi, 400)
            y_vals = action_lambda(x_vals) - 2

            # Plot the action and log to wandb
            if self.save_images:
                plt.plot(x_vals, y_vals, label="Action", color="b")
                im = wandb.Image(fig, caption="action")
                plt.tight_layout()
                os.makedirs("actions/", exist_ok=True)
                plt.savefig(f"actions/action_{info['t']}.svg", format="svg")
                plt.close(fig)
                wandb.log(
                    {
                        f"ep{episode_idx}/action": im,
                    }
                )

            # save values for video
            self.actions.append({"y": y_vals, "x": x_vals})

    def reset(self):
        # plot actions
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.set_theme()

        ax.set_xlabel("Spatial x")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(ticks=[0, np.pi, 2 * np.pi], labels=["0", r"$\pi$", r"$2\pi$"])

        ax.set_ylabel(r"Control Input $\hat{T}$")
        ax.set_ylim(-1, 1)
        ax.set_yticks(ticks=[-0.75, 0, 0.75], labels=["-C", "0", "C"])
        fig.tight_layout()

        artists = []
        for action in self.actions:
            artists.append(ax.plot(action["x"], action["y"], label="Action", color="b"))
        ani = animation.ArtistAnimation(fig=fig, artists=artists)
        writer = animation.FFMpegWriter(fps=2)
        path = pathlib.Path(f"{tempfile.gettempdir()}/rbcdata").resolve()
        path.mkdir(parents=True, exist_ok=True)
        path = f"{path}/actions.mp4"
        ani.save(path, writer=writer)

        # wandb
        if self.save_video:
            vid = wandb.Video(path, caption="actions")
            wandb.log({f"ep{self.ep_idx}/action_video": vid})
