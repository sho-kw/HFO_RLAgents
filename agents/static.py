import numpy as np
from .core import AbstractAgent


class _StaticAgent(AbstractAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_episode(self, *args):
        return self.select_best_action()

    def step(self, *args):
        return self.select_best_action()

    def end_episode(self, *args):
        pass


class RandomAgent(_StaticAgent):
    def __init__(self, action_space, *args, **kwargs):
        super().__init__(action_space, *args, **kwargs)
        if isinstance(action_space, int):
            self._random = lambda: self._random_int(action_space)
        else:
            raise NotImplementedError()

    def _random_int(self, high):
        return np.random.randint(0, high)

    def select_best_action(self, *args):
        return self._random()


class ConstantAgent(_StaticAgent):
    def __init__(self, action_space, constant_action, *args, **kwargs):
        super().__init__(action_space, *args, **kwargs)
        self.constant_action = constant_action

    def select_best_action(self, *args):
        return self.constant_action
