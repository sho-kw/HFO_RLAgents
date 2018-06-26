import numpy as np

class OU:
    def __init__(self, r, mu, theta, sigma, dt):
        self.r = r
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
    def __call__(self, v):
        self._update()
        return self.r + v
    def _update(self):
        self.r += self.theta * (self.mu - self.r) * self.dt + self.sigma * np.random.normal()
