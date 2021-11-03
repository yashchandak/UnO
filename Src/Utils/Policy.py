import numpy as np
import torch
from Src.Utils.utils import NeuralNet

from Src.Utils.Policy import Policy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torch import tensor, float32
from torch.distributions import Normal



def get_Policy(state_dim, config):
    if config.cont_actions:
        atype = torch.float32
        actor = Insulin_Gaussian(state_dim=state_dim, config=config)
        action_size = actor.action_dim
    else:
        atype = torch.long
        action_size = 1
        actor = Categorical(state_dim=state_dim, config=config)

    return actor, atype, action_size



class Policy(NeuralNet):
    def __init__(self, state_dim, config, action_dim=None):
        super(Policy, self).__init__()

        self.config = config
        self.state_dim = state_dim
        if action_dim is None:
            self.action_dim = config.env.action_space.shape[0]
        else:
            self.action_dim = action_dim

    def init(self):
        temp, param_list = [], []
        for name, param in self.named_parameters():
            temp.append((name, param.shape))
            if 'var' in name:
                param_list.append(
                    {'params': param, 'lr': self.config.actor_lr / 100})  # Keep learning rate of variance much lower
            else:
                param_list.append({'params': param})
        self.optim = self.config.optim(param_list, lr=self.config.actor_lr)

        print("Policy: ", temp)


class Categorical(Policy):
    def __init__(self, state_dim, config, action_dim=None):
        super(Categorical, self).__init__(state_dim, config)

        # overrides the action dim variable defined by super-class
        if action_dim is not None:
            self.action_dim = action_dim

        self.random = np.ones(self.action_dim)/self.action_dim

        self.fc1 = nn.Linear(self.state_dim, self.action_dim)
        self.init()

    def re_init_optim(self):
        self.optim = self.config.optim(self.parameters(), lr=self.config.actor_lr)

    def forward(self, state):
        x = self.fc1(state)
        return x

    def get_action_POMDP(self, state, state_POMDP, explore=0, behavior=False):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        probs = dist.cpu().view(-1).data.numpy()

        x_POMDP = self.forward(state_POMDP)
        dist_POMDP = F.softmax(x_POMDP, -1)
        probs_POMDP = dist_POMDP.cpu().view(-1).data.numpy()

        rho = 1
        if behavior:
            # Create behavior policy
            # By mixing evaluation policy and random policy
            new_probs = self.config.alpha * probs_POMDP +  (1-self.config.alpha) * self.random

            # Bug with numpy, floating point errors don't let prob to sum to 1 exactly
            # https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
            # solution is to normalize the probabilities by dividing them by their sum if the sum is close enough to 1
            new_probs /= new_probs.sum()
            action = np.random.choice(self.action_dim, p=new_probs)

            beta = new_probs[action]
            pi = probs_POMDP[action]
            rho = pi/beta
        else:
            action = np.random.choice(self.action_dim, p=probs_POMDP)

        return action, rho


    def get_action(self, state, explore=0, behavior=False):
        x = self.forward(state)
        dist = F.softmax(x, -1)
        probs = dist.cpu().view(-1).data.numpy()

        rho = 1
        if behavior:
            # Create behavior policy
            # By mixing evaluation policy and random policy
            new_probs = self.config.alpha * probs +  (1-self.config.alpha) * self.random

            # Bug with numpy, floating point errors don't let prob to sum to 1 exactly
            # https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
            # solution is to normalize the probabilities by dividing them by their sum if the sum is close enough to 1
            new_probs /= new_probs.sum()
            action = np.random.choice(self.action_dim, p=new_probs)

            beta = new_probs[action]
            pi = probs[action]
            rho = pi/beta
        else:
            action = np.random.choice(self.action_dim, p=probs)

        return action, rho

    def get_logprob_dist(self, state, action):
        x = self.forward(state)                                                              # BxA
        log_dist = F.log_softmax(x, -1)                                                      # BxA
        return log_dist.gather(1, action), log_dist                                          # BxAx(Bx1) -> B


