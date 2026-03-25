"""Random baseline policy for batting environment."""

import numpy as np


class RandomPolicy:
    """Samples random actions from the action space each step."""

    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        pass

    def get_action(self, obs):
        return self.action_space.sample()
