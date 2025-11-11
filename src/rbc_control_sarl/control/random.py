from .policy import Policy


class RandomPolicy(Policy):
    def call(self, obs):
        return self.action_space.sample()
