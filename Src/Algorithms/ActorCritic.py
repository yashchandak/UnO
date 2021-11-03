import numpy as np
import torch
from torch import tensor, float32
import torch.nn.functional as F
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, Critic, utils, Policy


class ActorCritic(Agent):
    def __init__(self, config):
        super(ActorCritic, self).__init__(config)
        # Get state features and instances for Actor and Value function
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = Policy.get_Policy(state_dim=self.state_features.feature_dim,
                                                                       config=config)
        self.critic = Critic.Critic(state_dim=self.state_features.feature_dim, config=config)
        self.trajectory = utils.Trajectory(max_len=self.config.batch_size, state_dim=self.state_dim,
                                           action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)

        self.modules = [('actor', self.actor), ('baseline', self.critic), ('state_features', self.state_features)]
        self.init()

    def get_action(self, state, behavior=False):
        explore = 0 # Don't do eps-greedy with policy gradients
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
        action, rho = self.actor.get_action(state, explore=0, behavior=behavior)

        # if self.config.debug:
        #     self.track_entropy(dist, action)
        return action, rho

    def get_action_POMDP(self, state, behavior=False):
        explore = 0 # Don't do eps-greedy with policy gradients
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = state.view(1, -1)
        
        # For the evaluation policy, make the state partially observable
        # i.e., over here we remove the second component of the state
        state_POMDP = state.detach().clone()
        state_POMDP[0, 1] = -1

        state = self.state_features.forward(state)
        state_POMDP = self.state_features.forward(state_POMDP)

        action, rho = self.actor.get_action_POMDP(state, state_POMDP, explore=0, behavior=behavior)

        # if self.config.debug:
        #     self.track_entropy(dist, action)
        return action, rho

    def update(self, s1, a1, _, r1, s2, done):
        self.trajectory.add(s1, a1, -1, r1, s2, int(done != 1))  # set dist=-1; Unused value

        if self.trajectory.size >= self.config.batch_size or done:
            self.optimize()
            self.trajectory.reset()

    def optimize(self):
        s1, a1, _, r1, s2, not_absorbing = self.trajectory.get_all()

        s1 = self.state_features.forward(s1)
        s2 = self.state_features.forward(s2)

        # ---------------------- optimize critic ----------------------
        next_val = self.critic.forward(s2).detach()    # Detach targets from grad computation.
        val_exp  = next_val * self.config.gamma * not_absorbing + r1
        val_pred = self.critic.forward(s1)

        # loss_critic = F.smooth_l1_loss(val_pred, val_exp)
        loss_critic = F.mse_loss(val_pred, val_exp)

        # ---------------------- optimize actor ----------------------
        td_error = (val_exp - val_pred).detach()
        logp, log_pi_all = self.actor.get_logprob_dist(s1, a1)
        loss_actor = -1.0 * torch.mean(td_error * logp)

        if self.config.entropy_lambda > 0:
            pi_all = torch.exp(log_pi_all)  # (BxH)xA -> BxHxA
            entropy = torch.mean(torch.sum(pi_all * log_pi_all, dim=-1))

            loss_actor = loss_actor + self.config.entropy_lambda * entropy

        self.step(loss_actor + loss_critic)#, clip_norm=5)


