from __future__ import print_function
import numpy as np
import torch
from torch import tensor, float32, int32
from torch.autograd import Variable
import torch.nn as nn
import shutil
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync
import importlib
from time import time
import sys
import pickle

np.random.seed(0)
torch.manual_seed(0)
dtype = torch.FloatTensor

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, restore, method):
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method

        if self.file:
            if restore:
                self.log = open(path.join(log_path, "logfile.log"), "a")
            else:
                self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        if self.file:
            self.log.flush()
            fsync(self.log.fileno())



def save_plots(rewards, config, name='rewards'):
    np.save(config.paths['results'] + name, rewards)
    if config.debug:
        if 'Grid' in config.env_name or 'room' in config.env_name:
            # Save the heatmap
            plt.figure()
            plt.title("Exploration Heatmap")
            plt.xlabel("100x position in x coordinate")
            plt.ylabel("100x position in y coordinate")
            plt.imshow(config.env.heatmap, cmap='hot', interpolation='nearest', origin='lower')
            plt.savefig(config.paths['results'] + 'heatmap.png')
            np.save(config.paths['results'] + "heatmap", config.env.heatmap)
            config.env.heatmap.fill(0)  # reset the heat map
            plt.close()

        plt.figure()
        plt.ylabel("Total return")
        plt.xlabel("Episode")
        plt.title("Performance")
        plt.plot(rewards)
        plt.savefig(config.paths['results'] + "performance.png")
        plt.close()


def plot(rewards):
    # Plot the results
    plt.figure(1)
    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel("Trajectories")
    plt.ylabel("Reward")
    plt.title("Baseline Reward")
    plt.show()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    def custom_weight_init(self):
        # Initialize the weight values
        for m in self.modules():
            weight_init(m)

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")

    def reset(self):
        return




def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def squash(x, eps = 1e-5):
    """
    Squashes each vector to ball of radius 1 - \eps

    :param x: (batch x dimension)
    :return: (batch x dimension)
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    unit = x / norm
    scale = norm**2/(1 + norm**2) - eps
    x = scale * unit

    # norm_2 = torch.sum(x**2, dim=-1, keepdim=True)
    # unit = x / torch.sqrt(norm_2)
    # scale = norm_2 / (1.0 + norm_2)    # scale \in [0, 1 - eps]
    # x = scale * unit - eps  # DO NOT DO THIS. it will make magnitude of vector consisting of all negatives larger

    return x


class Space:
    def __init__(self, low=[0], high=[1], dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size, )
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)

def get_var_w(shape, scale=1):
    w = torch.Tensor(shape[0], shape[1])
    w = nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('sigmoid'))
    return Variable(w.type(dtype), requires_grad=True)


def get_var_b(shape):
    return Variable(torch.rand(shape).type(dtype) / 100, requires_grad=True)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = 0#.1/ np.sqrt((fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        # m.weight.data.normal_(0.0, 0.03)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()




def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, name)
        if not exact and name in file:
            return path.join(dir, name)
    else:
        # recursive scan
        for file in all_files:
            if file == 'Experiments':
                continue
            _path = path.join(dir, file)
            if path.isdir(_path):
                location = search(_path, name, exact)
                if location:
                    return location

def dynamic_load(dir, name, load_class=False):
    try:
        # abs_path = search(dir, name).split('/')[1:]
        abs_path = search(dir, name).split('\\')[1:]
        pos = abs_path.index('HCODE-pvt')
        module_path = '.'.join([str(item) for item in abs_path[pos + 1:]])
        print("Module path: ", module_path, name)
        if load_class:
            obj = getattr(importlib.import_module(module_path), name)
        else:
            obj = importlib.import_module(module_path)
        print("Dynamically loaded from: ", obj)
        return obj
    except:
        raise ValueError("Failed to dynamically load the class: " + name )

def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    dir_path = str.split(dir_path, sep='/')[1:-1]  #Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join('/', *(dir_path[:i + 1])))


def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


def clip_norm(params, max_norm=1):
    # return params
    norm_param = []
    for param in params:
        norm = np.linalg.norm(param, 2)
        if norm > max_norm:
            norm_param.append(param/norm * max_norm)
        else:
            norm_param.append(param)
    return norm_param



class Trajectory:
    """
    Pre-allocated memory interface for storing and using on-policy trajectories

    Note: slight abuse of notation.
          sometimes Code treats 'dist' as extra variable and uses it to store other things, like: prob, etc.
    """
    def __init__(self, max_len, state_dim, action_dim, atype, config, dist_dim=1, stype=float32):

        self.s1 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
        self.a1 = torch.zeros((max_len, action_dim), dtype=atype, requires_grad=False, device=config.device)
        self.r1 = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
        self.s2 = torch.zeros((max_len, state_dim), dtype=stype, requires_grad=False, device=config.device)
        self.done = torch.zeros((max_len, 1), dtype=float32, requires_grad=False, device=config.device)
        self.dist = torch.zeros((max_len, dist_dim), dtype=float32, requires_grad=False, device=config.device)

        self.ctr = 0
        self.max_len = max_len
        self.atype = atype
        self.stype= stype
        self.config = config

    def add(self, s1, a1, dist, r1, s2, done):
        if self.ctr == self.max_len:
            # self.ctr = 0
            raise OverflowError

        self.s1[self.ctr] = torch.tensor(s1, dtype=self.stype)
        self.a1[self.ctr] = torch.tensor(a1, dtype=self.atype)
        self.dist[self.ctr] = torch.tensor(dist)
        self.r1[self.ctr] = torch.tensor(r1)
        self.s2[self.ctr] = torch.tensor(s2, dtype=self.stype)
        self.done[self.ctr] = torch.tensor(done)

        self.ctr += 1

    def reset(self):
        self.ctr = 0

    @property
    def size(self):
        return self.ctr

    def _get(self, ids):
        return self.s1[ids], self.a1[ids], self.dist[ids], self.r1[ids], self.s2[ids], self.done[ids]

    def get_current_transitions(self):
        pos = self.ctr
        return self.s1[:pos], self.a1[:pos], self.dist[:pos], self.r1[:pos], self.s2[:pos], self.done[:pos]

    def get_all(self):
        return self.s1, self.a1, self.dist, self.r1, self.s2, self.done

    def get_latest(self):
        return self._get([-1])

    def batch_sample(self, batch_size, nth_return):
        # Compute the estimated n-step gamma return
        R = nth_return
        for idx in range(self.ctr-1, -1, -1):
            R = self.r1[idx] + self.config.gamma * R
            self.r1[idx] = R

        # Genreate random sub-samples from the trajectory
        perm_indices = np.random.permutation(self.ctr)
        for ids in [perm_indices[i:i + batch_size] for i in range(0, self.ctr, batch_size)]:
            yield self._get(ids)
