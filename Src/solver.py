#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function
# from memory_profiler import profile


import argparse
from datetime import datetime

import numpy as np
import Src.Utils.utils as utils
from Src.config import Config
from time import time
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)


    def train(self, max_episodes):
        # Learn the model on the environment
        return_history = []
        true_rewards = []
        action_prob = []

        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0

        steps = 0
        t0 = time()
        for episode in range(start_ep, max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                action, dist = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)
                self.model.update(state, action, dist, reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                step += 1
                # if step >= self.config.max_steps:
                #     break

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            rm = 0.9*rm + 0.1*total_r
            # rm = total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                self.model.save()
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0


    def eval(self, max_episodes):
        self.model.load()
        temp = max_episodes/100

        returns = []
        for episode in range(max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                
                # IMPORTANT: Switch between the following two lines depending on
                # whether the domain is MDP or POMDP
                
                # action, dist = self.model.get_action(state)
                action, dist = self.model.get_action_POMDP(state)


                new_state, reward, done, info = self.env.step(action=action)
                state = new_state

                # Tracking intra-episode progress
                total_r += self.config.gamma**step * reward

                step += 1
                # if step >= self.config.max_steps:
                #     break

            returns.append(total_r)
            if episode % temp == 0:
                print("Eval Collected {}/{} :: Mean return {}".format(episode, max_episodes, np.mean(returns)))

                np.save(self.config.paths['results'] + 'eval_returns_' + str(self.config.alpha) + '_' + str(self.config.seed) , returns)


    def collect(self, max_episodes):
        self.model.load()
        temp = max_episodes/100

        trajectories = []
        for episode in range(max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            traj = []
            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')

                # IMPORTANT: Switch between the following two lines depending on
                # whether the domain is MDP or POMDP
                
                # action, rho = self.model.get_action(state, behavior=True)
                action, rho = self.model.get_action_POMDP(state, behavior=True)
                
                new_state, reward, done, info = self.env.step(action=action)
                state = new_state

                # Track importance ratio of current action, and current reward
                traj.append((rho, reward))

                step += 1
                # if step >= self.config.max_steps:
                #     break

            # Make the length of all trajectories the same.
            # Make rho = 1 and reward = 0, which corresponds to a self loop in the terminal state
            for i in range(step, self.env.max_horizon):
                traj.append((1, 0))

            trajectories.append(traj)

            if episode % temp == 0:
                print("Beta Collected {}/{}".format(episode, max_episodes))

                np.save(self.config.paths['results'] + 'beta_trajectories_' + str(self.config.alpha) + '_' + str(self.config.seed) , trajectories)
