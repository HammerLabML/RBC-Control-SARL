from .policy import Policy


class ZeroPolicy(Policy):
    def call(self, obs):
        return self.action_space.sample() * 0
