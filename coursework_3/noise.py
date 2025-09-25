import numpy as np

class OUNoise:
    def __init__(self, size, seed=0, mu=0., theta=0.15, sigma=0.2):
        np.random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.size)
        self.state += dx
        return self.state
