from __future__ import print_function
import numpy as np
from Src.Utils.utils import Space
import matplotlib.pyplot as plt



class Reco(object):
    def __init__(self,
                 debug=True):

        self.debug = debug
        self.n_max_actions = 5
        self.state_dim = 1

        self.max_horizon = 1
        self.min_reward = 0
        self.max_reward = 10

        # The state and action space of the domain.
        self.action_space = Space(size=self.n_max_actions)
        self.observation_space = Space(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.state = np.array([1])          # State is always 1

        # Time counter
        self.episode = 0

        # Reward associated with each arm
        rng = np.random.RandomState(1)
        self.amplitude = rng.rand(self.n_max_actions) * 10

        # Add noise of different variances to each arm
        rng = np.random.RandomState(0)
        self.stds = rng.rand(self.n_max_actions) * 0.1

        print("Reward Amplitudes: {} :: Avg {} ".format(self.amplitude, np.mean(self.amplitude)))

        self.reset()

    def seed(self, seed):
        self.seeding = seed

    def reset(self):
        return self.state

    def step(self, action):
        assert 0 <= action < self.n_max_actions

        all = self.amplitude + np.random.randn(self.n_max_actions) * self.stds
        all = np.clip(all, a_min=self.min_reward, a_max=self.max_reward)
        reward = all[action]

        self.episode += 1
        return self.state, reward, True, {'Max': np.max(all)}

    def get_rewards(self):
        return self.amplitude + np.random.randn(self.n_max_actions) * self.stds


if __name__=="__main__":
    # Plotting Agent
    rewards_list = []
    n_actions = 5
    epochs = 1000
    all_rewards = np.zeros((n_actions, epochs))
    env = Reco(debug=True)

    for a in range(n_actions):
        env.episode = 0
        for i in range(epochs):
            state = env.reset()
            _, r, _, _ = env.step(a)
            all_rewards[a][i] = r


