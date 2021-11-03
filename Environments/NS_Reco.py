from __future__ import print_function
import numpy as np
from Src.Utils.utils import Space, stablesoftmax
import matplotlib.pyplot as plt

"""
"""

class NS_Reco(object):
    def __init__(self,
                 speed=0,
                 oracle=-1,
                 debug=True):

        self.debug = debug
        self.n_max_actions = 5
        self.state_dim = 1
        self.max_horizon = 1
        self.speed = speed
        self.oracle = oracle

        # The state and action space of the domain.
        self.action_space = Space(size=self.n_max_actions)
        self.observation_space = Space(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.state = np.array([1])          # State is always 1

        # Time counter
        self.episode = 0

        # Reward associated with each arm is computed based on
        # sinusoidal wave of varying amplitude and frequency
        rng = np.random.RandomState(1)
        self.amplitude = rng.rand(self.n_max_actions)

        rng = np.random.RandomState(0)
        self.frequency = rng.rand(self.n_max_actions) * self.speed * 0.005

        # Add noise of different variances to each arm
        rng = np.random.RandomState(0)
        self.stds = rng.rand(self.n_max_actions) * 0.1

        if self.oracle >= 0:
            self.amplitude = self.amplitude * np.sin(self.oracle * self.frequency)
            self.speed = 0

        print("Reward Amplitudes: {} :: Avg {} ".format(self.amplitude, np.mean(self.amplitude)))

        self.reset()

    def seed(self, seed):
        self.seeding = seed

    def reset(self):
        return self.state

    def step(self, action):
        assert 0 <= action < self.n_max_actions
        reward = self.get_rewards()[action]

        # The episode always ends in one step.
        self.episode += 1
        return self.state, reward, True, {'Max': np.max(all)}

    def get_rewards(self):
        if self.speed == 0:
            return self.amplitude + np.random.randn(self.n_max_actions) * self.stds
        else:
            return self.amplitude * np.sin(self.episode * self.frequency) + np.random.randn(self.n_max_actions) * self.stds

    def get_true_rewards(self, episode):
        if self.speed == 0:
            return self.amplitude 
        else:
            return self.amplitude * np.sin(episode * self.frequency) 



def plot():
    # Plotting Agent
    rewards_list = []
    n_actions = 5
    epochs = 1000
    speed = 2
    all_rewards = np.zeros((n_actions, epochs))
    exact_rewards = np.zeros((n_actions, epochs))
    env = NS_Reco(speed=speed, debug=True)

    for a in range(n_actions):
        env.episode = 0
        for i in range(epochs):
            state = env.reset()
            _, r, _, _ = env.step(a)
            all_rewards[a][i] = r
            exact_rewards[a][i] = env.get_true_rewards(env.episode)[a]




    import matplotlib as mpl
    SMALL_SIZE = 20
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 26

    # mpl.style.use('seaborn')  # https://matplotlib.org/users/style_sheets.html

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title('Non-Stationary Domain (speed={})'.format(speed))
    # ax1.set_ylim(max_y)
    ax1.locator_params(nbins=5)
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    # plt.figure()


    colors = plt.cm.get_cmap('tab10', n_actions + 5)  # https://matplotlib.org/gallery/color/colormap_reference.html
    # ax1.set_prop_cycle(colors)

    for a in range(n_actions):
        ax1.plot(all_rewards[a], alpha=0.3, color=colors(a+5))
        ax1.plot(exact_rewards[a], label=str(a+1), color=colors(a+5), linewidth=2)

    # plt.title('Speed: {}'.format(speed))
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.plot(np.max(all_rewards + 0.05, axis=0), color='black')
    # plt.show()


    fig1.savefig('Recommender' + str(speed) + "_true.png", bbox_inches="tight")

    figLegend1 = plt.figure(figsize=(8, 0.5))
    l1 = plt.figlegend(*ax1.get_legend_handles_labels(), loc='upper center', fancybox=True, shadow=True, ncol=n_actions)
    for line in l1.get_lines():
        line.set_linewidth(5.0)
    figLegend1.savefig('item_legend.png', bbox_inches="tight")


def collect_data():

    speeds = [0, 1, 2]
    n_trials = 30
    n_eps = 1000

    beta_traj = []
    eval_traj = []
    for speed in speeds:
        env = NS_Reco(speed=speed)

        # Determine evaluation policy's prob's
        true_future_rewards = env.get_true_rewards(n_eps+1)
        eval_probs = stablesoftmax(true_future_rewards/0.05)
        print("eval probabilities", eval_probs)

        ##############
        # Collect Data
        ##############
        speed_traj = []
        speed_eval = []
        n_acts = env.n_max_actions
        all_acts = np.arange(n_acts)
        beta_probs = np.ones(n_acts) / n_acts        # Uniform random behavior policy


        # Collect data for the behavior policy
        for trial in range(n_trials):
            trials_traj = []
            env.episode = 0
            
            for i in range(n_eps):
                ep_traj = []
                _ = env.reset()
                done = False

                while not done:
                    action = np.random.choice(all_acts, p=beta_probs)
                    _, reward, done, _ = env.step(action)
                    rho = eval_probs[action]/beta_probs[action]
                    ep_traj.append((rho, reward))

                trials_traj.append(ep_traj)
            speed_traj.append(trials_traj)
        beta_traj.append(speed_traj)

        # Collect data for the evaluation policy
        for trial in range(n_eps):
            env.episode = n_eps + 1
            action = np.random.choice(all_acts, p=eval_probs)
            _, reward, _, _ = env.step(action)
            speed_eval.append(reward)
        eval_traj.append(speed_eval)


    np.save("../Experiments/NS_Reco/beta_trajectories_" + str(n_eps), beta_traj)
    np.save("../Experiments/NS_Reco/eval_returns_" + str(n_eps), eval_traj)


# collect_data()
plot()