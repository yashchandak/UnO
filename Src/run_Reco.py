

import argparse
from datetime import datetime
from Src.config import Config
from time import time
from Src.solver import Solver

###########################################
# Runner code
###########################################


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Parameters for Hyper-param sweep
        parser.add_argument("--base", default=0, help="Base counter for Hyper-param search", type=int)
        parser.add_argument("--inc", default=0, help="Increment counter for Hyper-param search", type=int)
        parser.add_argument("--hyper", default='default', help="Which Hyper param settings")
        parser.add_argument("--seed", default=0, help="seed for variance testing", type=int)

        # General parameters
        parser.add_argument("--save_count", default=100, help="Number of ckpts for saving results and model", type=int)
        parser.add_argument("--optim", default='rmsprop', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",
                            choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--restore", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model ckpts")
        parser.add_argument("--summary", default=True, type=self.str2bool,
                            help="--UNUSED-- Visual summary of various stats")
        parser.add_argument("--max_episodes", default=1000, help="maximum number of training episodes", type=int)
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)
        parser.add_argument("--swarm", default=False, help="Running on swarm?", type=self.str2bool)

        # Book-keeping parameters
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")
        parser.add_argument("--experiment", default='Data', help="Name of the experiment")

        self.Env_n_Agent_args(parser)  # Decide the Environment and the Agent
        self.Main_AC_args(parser)  # General Basis, Policy, Critic

        self.parser = parser

    def Env_n_Agent_args(self, parser):
        parser.add_argument("--algo_name", default='ActorCritic', help="Learning algorithm")
        parser.add_argument("--env_name", default='Reco', help="Environment to run the code")

    def Main_AC_args(self, parser):
        parser.add_argument("--alpha", default=0.5, help="Mixing ratio for behavior policy", type=float)
        parser.add_argument("--gamma", default=0.99, help="Discounting factor", type=float)
        parser.add_argument("--actor_lr", default=1e-2, help="Learning rate of actor", type=float)
        parser.add_argument("--state_lr", default=1e-3, help="Learning rate of state features", type=float)
        parser.add_argument("--critic_lr", default=1e-3, help="Learning rate of state features", type=float)
        parser.add_argument("--entropy_lambda", default=0, help="Learning rate of state features", type=float)
        parser.add_argument("--batch_size", default=1, help="Learning rate of state features", type=int)
        # parser.add_argument("--gauss_std", default=1.5, help="Variance for gaussian policy", type=float)
        parser.add_argument("--raw_basis", default=True, help="No basis fn.", type=self.str2bool)
        parser.add_argument("--NN_basis_dim", default='32', help="Shared Dimensions for Neural network layers")

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser



# @profile
def main(mode='train', inc=-1, hyper='default', base=-1):
    t = time()
    args = Parser().get_parser().parse_args()

    if inc >= 0 and hyper != 'default' and base >= 0:
        args.inc = inc
        args.hyper = hyper
        args.base = base

    config = Config(args)
    solver = Solver(config=config)

    # Training mode
    if mode == 'train':
        solver.train(max_episodes=config.max_episodes)

    elif mode == 'eval':
        solver.eval(max_episodes=int(1e5))

    elif mode == 'collectdata':
        solver.collect(max_episodes=int(2e7))

    else:
        return ValueError

    print("Total time taken: {}".format(time()-t))

if __name__ == "__main__":
        # import cProfile
        # cProfile.run('main()', sort='cumtime')
        # main(mode='train')
        main(mode='eval')
        main(mode='collectdata')

