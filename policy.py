import numpy as np

class Random:
    def __call__(self, scores):
        return np.random.randint(len(scores)) 

class Greedy:
    def __call__(self, scores):
        return np.argmax(scores)

class EpsilonGreedy(Greedy):
    def __init__(self, eps):
        self.eps = eps
    def __call__(self, scores):
        if np.random.uniform() < self.eps:
            return Random.__call__(self, scores)
        else:
            return Greedy.__call__(self, scores)

class Boltzmann:
    def __init__(self, t=1):
        self.t = t
    def __call__(self, scores):
        q = np.asarray(scores, dtype='float32')
        q -= np.max(q)
        q /= self.t
        eq = np.exp(q, out=q)
        probs = np.divide(eq, np.sum(eq), out=eq)
        action = np.random.choice(len(scores), p=probs)
        return action

GreedyPolicy = Greedy
EpsilonGreedyPolicy = EpsilonGreedy
BoltzmannPolicy = Boltzmann
