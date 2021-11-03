import numpy as np
import os
import yaml
import heapq
import matplotlib.pyplot as plt
import matplotlib as mpl
from random import random


SMALL_SIZE =  18
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

# mpl.style.use('seaborn')  # https://matplotlib.org/users/style_sheets.html

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['svg.hashsalt'] = 1
np.random.seed(2)


# colors = plt.cm.get_cmap('tab10', len(compare_list))  # https://matplotlib.org/gallery/color/colormap_reference.html

# Just as reminder for future
domain_props = {
    'Reco': {'x_scale': 10, 'max_x': 1000, 'max_y': [0, 0.32], 'title': 'Recommender System',
             'dir': 'final_data/30trials/Reco_'},

    'NS_Reco': {'x_scale': 10, 'max_x': 1000, 'max_y': [0, 0.32], 'title': 'Non-Stationary Domain',
             'dir': 'final_data/30trials/NS_Reco_'},

    'Diabetes': {'x_scale': 10, 'max_x': 1000, 'max_y': [0.25, 0.85], 'title': 'Diabetes Treatment',
           'dir': 'final_data/30trials/Diabetes_'},
        
    'Maze_POMDP': {'x_scale': 10, 'max_x': 1000, 'max_y': [0.35, 1.07], 'title': 'Gridworld (POMDP)',
        'dir': 'final_data/30trials/Maze_POMDP_'}

}


def average_plot(xs, ys, ax_handle, color, linewidth=2, alpha=0.2, label=""):
    mu, stderr = np.mean(ys, axis=0), 2*np.std(ys, axis=0)/np.sqrt(ys.shape[0])
    # mu, stderr = np.mean(ys, axis=0), np.std(ys, axis=0)/np.sqrt(ys.shape[0])
    if label != "":
        ax_handle.plot(xs, mu, label=label, c=color, linewidth=linewidth)
    else:
        ax_handle.plot(xs, mu, c=color, linewidth=linewidth)    
    ax_handle.fill_between(xs, mu - stderr, mu + stderr, alpha=alpha, facecolor=color)



def plot(domain):
    # Load all the paths
    path = domain_props[domain]['dir']

    CDF_estimates = np.load(path + 'CDF.npy')
    mean_estimates = np.load(path + 'mean.npy')
    variance_estimates = np.load(path + 'variance.npy')
    quantile_estimates = np.load(path + 'quantile.npy')
    cvar_estimates = np.load(path + 'cvar.npy')

    CDF_bounds = np.load(path + 'CDF_bounds.npy')
    mean_bounds = np.load(path + 'mean_bounds.npy')
    mean_baseline_bounds = np.load(path + 'mean_baseline_bounds.npy')
    variance_bounds = np.load(path + 'variance_bounds.npy')
    variance_baseline_bounds = np.load(path + 'variance_baseline_bounds.npy')
    quantile_bounds = np.load(path + 'quantile_bounds.npy')
    cvar_bounds = np.load(path + 'cvar_bounds.npy')

    CDF_boot_bounds = np.load(path + 'CDF_boot_bounds.npy')
    mean_boot_bounds = np.load(path + 'mean_boot_bounds.npy')
    variance_boot_bounds = np.load(path + 'variance_boot_bounds.npy')
    quantile_boot_bounds = np.load(path + 'quantile_boot_bounds.npy')
    cvar_boot_bounds = np.load(path + 'cvar_boot_bounds.npy')

    true_CDF = np.load(path + 'true_CDF.npy')
    CDF_plot_xs = np.load(path + 'CDF_plot_xs.npy')
    true_params = np.load(path + 'true_params.npy')
    trend_xs = np.load(path + 'trend_xs.npy')

    print(trend_xs)

    ######## Start plotting ##########

    fig1 = plt.figure(figsize=(8, 25))

    ##### CDF Plots ########
    selected = len(trend_xs) - 1
    plotCDF = fig1.add_subplot(5, 1, 1)
    plotCDF.set_title(domain_props[domain]['title'])
    plotCDF.set_xlabel("Return")
    # plotCDF.set_ylabel("CDF")
    plotCDF.ticklabel_format(style='plain', axis='x', scilimits=(0, 0), useMathText=True)
    plotCDF.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
    plotCDF.plot(CDF_plot_xs, true_CDF, linestyle='--', linewidth=3, color='black')
    average_plot(CDF_plot_xs, CDF_estimates[:, selected, :], plotCDF, color='green')
    average_plot(CDF_plot_xs, CDF_bounds[:, selected, 0, :], plotCDF, color='red')
    average_plot(CDF_plot_xs, CDF_bounds[:, selected, 1, :], plotCDF, color='red')
    
    ###### Plot Mean #########
    plotMean = fig1.add_subplot(5, 1, 2)
    plotMean.set_xticklabels([])
    if domain == 'Reco': plotMean.set_ylim([5.0, 9.0])
    if domain == 'Maze_POMDP': plotMean.set_ylim([-6, -1])
    if domain == 'Diabetes': plotMean.set_ylim([-10.0, 15.0])
    # plotMean.set_ylabel("Mean")
    plotMean.hlines(true_params[0], xmin=min(trend_xs), xmax=max(trend_xs), linestyle='--', linewidth=3, color='black', label="True")
    average_plot(trend_xs, mean_estimates[:, :], plotMean, color='green', label="UnO")
    average_plot(trend_xs, mean_bounds[:,:,0], plotMean, color='red')
    average_plot(trend_xs, mean_bounds[:,:,1], plotMean, color='red', label="UnO-CI")
    average_plot(trend_xs, mean_boot_bounds[:,:,0], plotMean, color='blue')
    average_plot(trend_xs, mean_boot_bounds[:,:,1], plotMean, color='blue', label="UnO-Boot")
    average_plot(trend_xs, mean_baseline_bounds[:,:,0], plotMean, color='orange')
    average_plot(trend_xs, mean_baseline_bounds[:,:,1], plotMean, color='orange', label="Baseline-CI")
    plt.xscale("log")

    ###### Plot Variance #########
    plotVariance = fig1.add_subplot(5, 1, 3)
    plotVariance.set_xticklabels([])
    if domain == 'Reco': plotVariance.set_ylim([-5, 30])
    if domain == 'Maze_POMDP': plotVariance.set_ylim([-0.5, 7.0])
    if domain == 'Diabetes': plotVariance.set_ylim([0.0, 210.0])
    # plotVariance.set_ylabel("Variance")
    plotVariance.hlines(true_params[3], xmin=min(trend_xs), xmax=max(trend_xs), linestyle='--', linewidth=3, color='black', label="True")
    average_plot(trend_xs, variance_estimates[:, :], plotVariance, color='green', label="UnO")
    average_plot(trend_xs, variance_bounds[:,:,0], plotVariance, color='red')
    average_plot(trend_xs, variance_bounds[:,:,1], plotVariance, color='red', label="UnO-CI")
    average_plot(trend_xs, variance_boot_bounds[:,:,0], plotVariance, color='blue')
    average_plot(trend_xs, variance_boot_bounds[:,:,1], plotVariance, color='blue', label="UnO-Boot")
    average_plot(trend_xs, variance_baseline_bounds[:,:,0], plotVariance, color='orange')
    average_plot(trend_xs, variance_baseline_bounds[:,:,1], plotVariance, color='orange', label="Baseline-CI")
    plt.xscale("log")


    ###### Plot Quantiles #########
    plotQuantile = fig1.add_subplot(5, 1, 4)
    plotQuantile.set_xticklabels([])
    if domain == 'Reco': plotQuantile.set_ylim([5, 9])
    if domain == 'Maze_POMDP': plotQuantile.set_ylim([-5, -1])
    if domain == 'Diabetes': plotQuantile.set_ylim([5.0, 16.0])
    # plotQuantile.set_ylabel("Quantile (0.5)")
    plotQuantile.hlines(true_params[1], xmin=min(trend_xs), xmax=max(trend_xs), linestyle='--', color='black', linewidth=3)
    average_plot(trend_xs, quantile_estimates[:, :], plotQuantile, color='green')
    average_plot(trend_xs, quantile_bounds[:,:,0], plotQuantile, color='red')
    average_plot(trend_xs, quantile_bounds[:,:,1], plotQuantile, color='red')
    average_plot(trend_xs, quantile_boot_bounds[:,:,0], plotQuantile, color='blue')
    average_plot(trend_xs, quantile_boot_bounds[:,:,1], plotQuantile, color='blue')
    plt.xscale("log")


    ##### CVAR Plots ########
    plotCVaR = fig1.add_subplot(5, 1, 5)
    # plotCVaR.set_xticklabels([])
    plotCVaR.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plotCVaR.set_xlabel("Number of Trajectories")
    # plotCVaR.set_ylabel("CVaR (0.25)")
    if domain == 'Reco': plotCVaR.set_ylim([2,8.0])
    if domain == 'Maze_POMDP': plotCVaR.set_ylim([-6, -1])
    if domain == 'Diabetes': plotCVaR.set_ylim([-16.0, 10.0])
    plotCVaR.hlines(true_params[2], xmin=min(trend_xs), xmax=max(trend_xs), linestyle='--', color='black', linewidth=3)
    average_plot(trend_xs, cvar_estimates[:, :], plotCVaR, color='green')
    average_plot(trend_xs, cvar_bounds[:,:,0], plotCVaR, color='red')
    average_plot(trend_xs, cvar_bounds[:,:,1], plotCVaR, color='red')
    average_plot(trend_xs, cvar_boot_bounds[:,:,0], plotCVaR, color='blue')
    average_plot(trend_xs, cvar_boot_bounds[:,:,1], plotCVaR, color='blue')

    plt.xscale("log")


    ## Legends
    figLegend1 = plt.figure(figsize=(8, 0.5))
    l1 = plt.figlegend(*plotMean.get_legend_handles_labels(), loc='upper center', fancybox=True, shadow=True, ncol=6)
    for line in l1.get_lines():
        line.set_linewidth(5.0)
    figLegend1.savefig('legend.png', bbox_inches="tight")

    # fig1.tight_layout()
    fig1.savefig(domain + "_bounds.png", bbox_inches="tight")
    plt.show()



def NS_plot(domain):
    # Load all the paths
    path = domain_props[domain]['dir']

    CDF_bounds = np.load(path + 'CDF_bounds_8.npy')
    CDF_bounds_0 = np.load(path + 'CDF_bounds_0.npy')
    true_CDF = np.load(path + 'true_CDF.npy')
    CDF_plot_xs = np.load(path + 'CDF_plot_xs.npy')

    ######## Start plotting ##########

    # fig1 = plt.figure(figsize=(30,5))
    fig1 = plt.figure(figsize=(8, 6))

    ##### CDF Plots ########
    # plotCDF1 = fig1.add_subplot(1, 3, 1)
    plotCDF1 = fig1.add_subplot(1, 1, 1)
    plotCDF1.locator_params(axis='x', nbins=5)
    plotCDF1.set_title(domain_props[domain]['title']+'(Speed=0)')
    plotCDF1.set_xlabel("Return")
    plotCDF1.set_ylabel("CDF")
    plotCDF1.ticklabel_format(style='plain', axis='x', scilimits=(0, 0), useMathText=True)
    plotCDF1.plot(CDF_plot_xs, true_CDF[0, :], linestyle='--', linewidth=3, color='black', label='True')
    average_plot(CDF_plot_xs, CDF_bounds[0, :, 0, :], plotCDF1, color='blue')
    average_plot(CDF_plot_xs, CDF_bounds[0, :, 1, :], plotCDF1, color='blue', label='UnO-NonStationary')
    average_plot(CDF_plot_xs, CDF_bounds_0[0, :, 0, :], plotCDF1, color='red')
    average_plot(CDF_plot_xs, CDF_bounds_0[0, :, 1, :], plotCDF1, color='red', label='UnO')

    fig1.savefig(domain + "_bounds0.png", bbox_inches="tight")
    plt.clf()

    # plotCDF2 = fig1.add_subplot(1, 3, 2)
    plotCDF2 = fig1.add_subplot(1, 1, 1)
    plotCDF2.locator_params(axis='x', nbins=5)
    plotCDF2.set_title(domain_props[domain]['title']+'(Speed=1)')
    plotCDF2.set_xlabel("Return")
    plotCDF2.set_ylabel("CDF")
    plotCDF2.ticklabel_format(style='plain', axis='x', scilimits=(0, 0), useMathText=True)
    plotCDF2.plot(CDF_plot_xs, true_CDF[1, :], linestyle='--', linewidth=3, color='black', label='True')
    average_plot(CDF_plot_xs, CDF_bounds[1, :, 0, :], plotCDF2, color='blue')
    average_plot(CDF_plot_xs, CDF_bounds[1, :, 1, :], plotCDF2, color='blue', label='UnO-NonStationary')
    average_plot(CDF_plot_xs, CDF_bounds_0[1, :, 0, :], plotCDF2, color='red')
    average_plot(CDF_plot_xs, CDF_bounds_0[1, :, 1, :], plotCDF2, color='red', label='UnO')

    fig1.savefig(domain + "_bounds1.png", bbox_inches="tight")
    plt.clf()

    # plotCDF3 = fig1.add_subplot(1, 3, 3)
    plotCDF3 = fig1.add_subplot(1, 1, 1)
    plotCDF3.locator_params(axis='x', nbins=5)
    plotCDF3.set_title(domain_props[domain]['title']+'(Speed=2)')
    plotCDF3.set_xlabel("Return")
    plotCDF3.set_ylabel("CDF")
    plotCDF3.ticklabel_format(style='plain', axis='x', scilimits=(0, 0), useMathText=True)
    plotCDF3.plot(CDF_plot_xs, true_CDF[2, :], linestyle='--', linewidth=3, color='black', label='True')
    average_plot(CDF_plot_xs, CDF_bounds[2, :, 0, :], plotCDF3, color='blue')
    average_plot(CDF_plot_xs, CDF_bounds[2, :, 1, :], plotCDF3, color='blue', label='UnO-NonStationary')
    average_plot(CDF_plot_xs, CDF_bounds_0[2, :, 0, :], plotCDF3, color='red')
    average_plot(CDF_plot_xs, CDF_bounds_0[2, :, 1, :], plotCDF3, color='red', label='UnO')

    figLegend1 = plt.figure(figsize=(8, 0.5))
    l1 = plt.figlegend(*plotCDF3.get_legend_handles_labels(), loc='upper center', fancybox=True, shadow=True, ncol=6)
    for line in l1.get_lines():
        line.set_linewidth(5.0)
    figLegend1.savefig('NS_legend.png', bbox_inches="tight")

    fig1.savefig(domain + "_bounds2.png", bbox_inches="tight")
    # fig1.savefig(domain + "_bounds.png", bbox_inches="tight")
    plt.show()




# plot("Reco")
plot("Diabetes")
# plot("Maze_POMDP")

# NS_plot("NS_Reco")