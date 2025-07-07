from stable_baselines3.common.callbacks import BaseCallback


class NusseltCallbackSB3(BaseCallback):
    def __init__(
        self,
        freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        for info in infos:
            self.logger.record_mean("rollout/nusselt_obs_mean", info["nusselt_obs"])
            self.logger.record_mean("rollout/nusselt_mean", info["nusselt_state"])
            # TODO self.logger.record_mean("rollout/cell_dist_mean", info["cell_dist"])
        return True
