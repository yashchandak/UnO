
using Plots
using NPZ
using Random

include("../HCOPE.jl")
include("../HCOVE.jl")
include("../UnO_utils.jl")
include("../UnO_params.jl")
include("../UnO_CDF.jl")


function compute_trend(props,
                        data::Array{Float64, 3},
                        pi_data::Array{Float64, 1},
                        trials::Int,
                        mini::Float64,
                        maxi::Float64,
                        intervals::Int)

    ######################################################
    # Pre-allocate arrays for storing all the estimates
    n_points = 1000      # Number of support points for plotting CDFs
    α_quantile = 0.5     # For inverse CDF based stuff
    α_cvar = 0.25    # For inverse CDF based stuff
    δ = 0.05             # Failure rates

    CDF_estimates = fill(0.0, trials, intervals, n_points)
    mean_estimates = fill(0.0, trials, intervals)
    variance_estimates = fill(0.0, trials, intervals)
    quantile_estimates = fill(0.0, trials, intervals)
    cvar_estimates = fill(0.0, trials, intervals)

    CDF_bounds = fill(0.0, trials, intervals, 2, n_points)
    mean_bounds = fill(0.0, trials, intervals, 2)
    mean_baseline_bounds = fill(0.0, trials, intervals, 4)
    variance_bounds = fill(0.0, trials, intervals, 2)
    variance_baseline_bounds = fill(0.0, trials, intervals, 4)
    quantile_bounds = fill(0.0, trials, intervals, 2)
    cvar_bounds = fill(0.0, trials, intervals, 2)

    CDF_boot_bounds = fill(0.0, trials, intervals, 2, n_points)
    mean_boot_bounds = fill(0.0, trials, intervals, 2)
    variance_boot_bounds = fill(0.0, trials, intervals, 2)
    quantile_boot_bounds = fill(0.0, trials, intervals, 2)
    cvar_boot_bounds = fill(0.0, trials, intervals, 2)

    # True CDF and parameters
    true_CDF = fill(0.0, n_points)
    CDF_plot_xs = fill(0.0, n_points)
    true_params = fill(0.0, 5)      # Order: Mean, quantile, CVaR, variance, entropy

    ###################################################
    # Compute estimates and bounds for CDF and params for different trials
    ###################################################

    step = (maxi - mini) / (intervals - 1)
    full_size, _, _ = size(data)
    trend_xs = fill(0.0, intervals)

    # Run in CLI before executing code
    # Be careful, this changes arrays asynchronously
    # export JULIA_NUM_THREADS=10
    #
    for index in 1:1:intervals

        n = Int(floor( 10^(mini + (index - 1)*step) ))
        trend_xs[index] = n

        perm_idxs = shuffle(collect(1:full_size))   # Shuffle indices
        Threads.@threads for trial in 1:1:trials
            println(n, ":", trial)

            ## Without replacement sampling
            ## Ensure unique data for each trial
            # start = (trial-1)*n + 1
            # stop = n*trial
            # idxs = perm_idxs[start:stop]

            # With replacement sampling
            perm2 = shuffle(perm_idxs)
            idxs = perm2[1:n]

            fragment = data[idxs, :, :]

            ###########################################
            # Estimators for CDF and other parameters
            ###########################################

            xs, ys = IS_CDF(fragment, props)
            wis_xs, wis_ys = WIS_CDF(fragment, props)     # WIS version

            CDF_estimates[trial, index, :] = get_equispaced_CDF(xs, ys, props, n_points)[2]
            mean_estimates[trial, index] = CDF_mean(xs, ys)
            variance_estimates[trial, index] = CDF_variance(xs, ys)
            quantile_estimates[trial, index] = CDF_quantile(xs, ys, α_quantile)
            cvar_estimates[trial, index] = CDF_CVaR(xs, ys, α_cvar)

            ############################################################
            # Guaranteed coverage bounds for CDF and other parameters
            ###########################################################

            # bound_xs, lb_ys, ub_ys = CDF_band(xs, ys, props, δ)
            bound_xs, lb_ys, ub_ys = CDF_band_optimized(xs, ys, props, δ)
            CDF_bounds[trial, index, 1, :] = get_equispaced_CDF(bound_xs, lb_ys, props, n_points)[2]
            CDF_bounds[trial, index, 2, :] = get_equispaced_CDF(bound_xs, ub_ys, props, n_points)[2]

            lb, ub = CDF_mean_bound(bound_xs, lb_ys, ub_ys)
            mean_bounds[trial, index, 1] = lb
            mean_bounds[trial, index, 2] = ub

            # Divide by 8 because of two-sided bounds for 4 statistics
            mean_baseline_bounds[trial, index, 1] = CI_lower(fragment, props, δ/8)
            mean_baseline_bounds[trial, index, 2] = CI_upper(fragment, props, δ/8)
            # mean_baseline_bounds[trial, index, 3] = CI_lower(fragment, props, δ/2)  # w/o union
            # mean_baseline_bounds[trial, index, 4] = CI_upper(fragment, props, δ/2)  # w/o union

            lb, ub = CDF_variance_bound(bound_xs, lb_ys, ub_ys)
            variance_bounds[trial, index, 1] = lb
            variance_bounds[trial, index, 2] = ub

            # Divide by 8 because of two-sided bounds for 4 statistics
            variance_baseline_bounds[trial, index, 1] = HCOVE_lower(fragment, props, δ/8)
            variance_baseline_bounds[trial, index, 2] = HCOVE_upper(fragment, props, δ/8)
            # variance_baseline_bounds[trial, index, 3] = HCOVE_lower(fragment, props, δ/2)   # w/o union
            # variance_baseline_bounds[trial, index, 4] = HCOVE_upper(fragment, props, δ/2)   # w/o union

            lb, ub = CDF_quantile_bound(bound_xs, lb_ys, ub_ys, α_quantile)
            quantile_bounds[trial, index, 1] = lb
            quantile_bounds[trial, index, 2] = ub

            lb, ub = CDF_CVaR_bound(bound_xs, lb_ys, ub_ys, α_cvar)
            cvar_bounds[trial, index, 1] = lb
            cvar_bounds[trial, index, 2] = ub

            ############################
            # Bootstrap based bounds
            ############################
            lb, ub = CDF_mean_boot_bound(wis_xs, wis_ys, δ/4)
            mean_boot_bounds[trial, index, 1] = lb
            mean_boot_bounds[trial, index, 2] = ub

            lb, ub = CDF_variance_boot_bound(wis_xs, wis_ys, δ/4)
            variance_boot_bounds[trial, index, 1] = lb
            variance_boot_bounds[trial, index, 2] = ub

            lb, ub = CDF_quantile_boot_bound(wis_xs, wis_ys, α_quantile, δ/4)
            quantile_boot_bounds[trial, index, 1] = lb
            quantile_boot_bounds[trial, index, 2] = ub

            lb, ub = CDF_CVaR_boot_bound(wis_xs, wis_ys, α_cvar, δ/4)
            cvar_boot_bounds[trial, index, 1] = lb
            cvar_boot_bounds[trial, index, 2] = ub

        end;
    end;


    #######################
    # True CDF and parameters
    ########################

    xs, ys = on_policy_CDF(pi_data)
    CDF_plot_xs, true_CDF = get_equispaced_CDF(xs, ys, props, n_points)
    true_params[1] = CDF_mean(xs, ys)
    true_params[2] = CDF_quantile(xs, ys, α_quantile)
    true_params[3] = CDF_CVaR(xs, ys, α_cvar)
    true_params[4] = CDF_variance(xs, ys)


    ##############################################
    ## Save all the results from the experiments
    #################################################

    npzwrite(props["name"]*"_CDF.npy", CDF_estimates)
    npzwrite(props["name"]*"_mean.npy", mean_estimates)
    npzwrite(props["name"]*"_variance.npy", variance_estimates)
    npzwrite(props["name"]*"_quantile.npy", quantile_estimates)
    npzwrite(props["name"]*"_cvar.npy", cvar_estimates)

    npzwrite(props["name"]*"_CDF_bounds.npy", CDF_bounds)
    npzwrite(props["name"]*"_mean_bounds.npy", mean_bounds)
    npzwrite(props["name"]*"_mean_baseline_bounds.npy", mean_baseline_bounds)
    npzwrite(props["name"]*"_variance_bounds.npy", variance_bounds)
    npzwrite(props["name"]*"_variance_baseline_bounds.npy", variance_baseline_bounds)
    npzwrite(props["name"]*"_quantile_bounds.npy", quantile_bounds)
    npzwrite(props["name"]*"_cvar_bounds.npy", cvar_bounds)

    npzwrite(props["name"]*"_CDF_boot_bounds.npy", CDF_boot_bounds)
    npzwrite(props["name"]*"_mean_boot_bounds.npy", mean_boot_bounds)
    npzwrite(props["name"]*"_variance_boot_bounds.npy", variance_boot_bounds)
    npzwrite(props["name"]*"_quantile_boot_bounds.npy", quantile_boot_bounds)
    npzwrite(props["name"]*"_cvar_boot_bounds.npy", cvar_boot_bounds)

    npzwrite(props["name"]*"_true_CDF.npy", true_CDF)
    npzwrite(props["name"]*"_CDF_plot_xs.npy", CDF_plot_xs)
    npzwrite(props["name"]*"_true_params.npy", true_params)
    npzwrite(props["name"]*"_trend_xs.npy", trend_xs)


    #######################
    # Plots
    ########################

    function average_plot!(xs, ys, label)
        μs = dropdims(mean(ys, dims=1), dims=1)
        σs = dropdims(std(ys, dims=1), dims=1) ./ sqrt(trials)
        plot!(xs,μs, ribbon=σs, fillalpha=.3, label=label, legend=:left)
    end

    ########## CDF Plots ############

    # selected_int = ceil(Int32, intervals/2 + 1)         # Take the middle number of samples
    selected_int = intervals         # Take the middle number of samples
    plotCDF = plot(CDF_plot_xs, true_CDF, label="Fπ")                   # true CDF
    average_plot!(CDF_plot_xs, CDF_estimates[:, selected_int, :], "̂̂F")              # esimtated CDF
    average_plot!(CDF_plot_xs, CDF_bounds[:, selected_int, 1, :], "̂̂F-")             # lower bound CDF
    average_plot!(CDF_plot_xs, CDF_bounds[:, selected_int, 2, :], "̂̂F+")             # upper bound CDF
    average_plot!(CDF_plot_xs, CDF_boot_bounds[:, selected_int, 1, :], "Boot-")     # boot lower bound CDF
    average_plot!(CDF_plot_xs, CDF_boot_bounds[:, selected_int, 2, :], "Boot+")     # boot upper bound CDF

    ######## Mean Plots ###############

    plotMean = plot([true_params[1]], linetype=:hline, label="μπ", xaxis=:log)   # true mean
    average_plot!(trend_xs, mean_estimates[:, :], "̂μ")                              # esimtated mean
    average_plot!(trend_xs, mean_bounds[:, :, 1], "̂μ-")                             # lower bound mean
    average_plot!(trend_xs, mean_bounds[:, :, 2], "̂μ+")                             # upper bound mean
    average_plot!(trend_xs, mean_baseline_bounds[:, :, 1], "Base-")                 # Baseline lower bound mean
    average_plot!(trend_xs, mean_baseline_bounds[:, :, 2], "Base+")                 # Baseline upper bound mean
    average_plot!(trend_xs, mean_boot_bounds[:, :, 1], "Boot-")                     # boot lower bound mean
    average_plot!(trend_xs, mean_boot_bounds[:, :, 2], "Boot+")                     # boot upper bound mean

    ######## Variance Plots ###############

    plotVariance = plot([true_params[4]], linetype=:hline, label="σπ", xaxis=:log)   # true mean
    average_plot!(trend_xs, variance_estimates[:, :], "̂μ")                              # esimtated mean
    average_plot!(trend_xs, variance_bounds[:, :, 1], "̂μ-")                             # lower bound mean
    average_plot!(trend_xs, variance_bounds[:, :, 2], "̂μ+")                             # upper bound mean
    average_plot!(trend_xs, variance_baseline_bounds[:, :, 1], "Base-")                 # Baseline lower bound mean
    average_plot!(trend_xs, variance_baseline_bounds[:, :, 2], "Base+")                 # Baseline upper bound mean
    average_plot!(trend_xs, variance_boot_bounds[:, :, 1], "Boot-")                     # boot lower bound mean
    average_plot!(trend_xs, variance_boot_bounds[:, :, 2], "Boot+")                     # boot upper bound mean

    ######## Quantile Plots ###############

    plotQuantile = plot([true_params[2]],linetype=:hline, label="Qπ", xaxis=:log)   # true quantile
    average_plot!(trend_xs, quantile_estimates[:, :], "̂Qπ")                             # esimtated quantile
    average_plot!(trend_xs, quantile_bounds[:, :, 1], "̂Q-")                             # lower bound quantile
    average_plot!(trend_xs, quantile_bounds[:, :, 2], "̂Q+")                             # upper bound quantile
    average_plot!(trend_xs, quantile_boot_bounds[:, :, 1], "Boot-")                     # boot lower bound quantile
    average_plot!(trend_xs, quantile_boot_bounds[:, :, 2], "Boot+")                     # boot upper bound quantile

    ######## CVaR Plots ###############

    plotCVaR = plot([true_params[3]],linetype=:hline, label="Cπ", xaxis=:log)   # true cvar
    average_plot!(trend_xs, cvar_estimates[:, :], "̂Cπ")                             # esimtated cvar
    average_plot!(trend_xs, cvar_bounds[:, :, 1], "̂C-")                             # lower bound cvar
    average_plot!(trend_xs, cvar_bounds[:, :, 2], "̂C+")                             # upper bound cvar
    average_plot!(trend_xs, cvar_boot_bounds[:, :, 1], "Boot-")                     # boot lower bound cvar
    average_plot!(trend_xs, cvar_boot_bounds[:, :, 2], "Boot+")                     # boot upper bound cvar

    ####### Failure rates ##############

    plotFailure = plot([δ], linetype=:hline, label="δ", xaxis=:log)

    # Simultaneous guaranteed coverage bounds failure
    f1l = mean_bounds[:, :, 1] .> true_params[1]         # Lower mean bound failed
    f2l = quantile_bounds[:, :, 1] .> true_params[2]     # Lower quantile bound failed
    f3l = cvar_bounds[:, :, 1] .> true_params[3]         # Lower cvar bound failed
    f4l = variance_bounds[:, :, 1] .> true_params[4]     # Lower variance bound failed

    f1u = mean_bounds[:, :, 2] .< true_params[1]         # Upper mean bound failed
    f2u = quantile_bounds[:, :, 2] .< true_params[2]     # Upper quantile bound failed
    f3u = cvar_bounds[:, :, 2] .< true_params[3]         # Upper cvar bound failed
    f4u = variance_bounds[:, :, 2] .< true_params[4]     # Upper variance bound failed

    ci_failure = f1l .| f2l .| f3l .| f4l .| f5l .| f1u .| f2u .| f3u .| f4u .| f5u       # Bitwise OR
    average_plot!(trend_xs, ci_failure, "CI")

    # Simultaneous Bootstrap bounds failure
    f1l = mean_boot_bounds[:, :, 1] .> true_params[1]         # Lower mean boot bound failed
    f2l = quantile_boot_bounds[:, :, 1] .> true_params[2]     # Lower quantile boot bound failed
    f3l = cvar_boot_bounds[:, :, 1] .> true_params[3]         # Lower cvar boot bound failed
    f4l = variance_boot_bounds[:, :, 1] .> true_params[4]     # Lower variance boot bound failed

    f1u = mean_boot_bounds[:, :, 2] .< true_params[1]         # Upper mean boot bound failed
    f2u = quantile_boot_bounds[:, :, 2] .< true_params[2]     # Upper quantile boot bound failed
    f3u = cvar_boot_bounds[:, :, 2] .< true_params[3]         # Upper cvar boot bound failed
    f4u = variance_boot_bounds[:, :, 2] .< true_params[4]     # Upper variance boot bound failed

    boot_failure = f1l .| f2l .| f3l .| f4l .| f1u .| f2u .| f3u .| f4u          # Bitwise OR
    average_plot!(trend_xs, boot_failure, "Boot")

    npzwrite(props["name"]*"_ci_failure.npy", ci_failure)
    npzwrite(props["name"]*"_boot_failure.npy", boot_failure)

    ###### Plot everything in a single plot ##########
    plot(plotCDF, plotMean, plotVariance, plotQuantile, plotCVaR, plotFailure, layout=(7,1))

end


function compute_NS_trend(props,
    data::Array{Float64, 5},  # speed, trial, number of eps, horizon, 2
    pi_data::Array{Float64, 2}, # speed, trials
    trials::Int)

    ######################################################
    # Pre-allocate arrays for storing all the estimates
    n_points = 1000      # Number of support points for plotting CDFs
    δ = 0.05             # Failure rates
    speeds = [0,1,2]     # Speeds of non-stationarity

    CDF_bounds = fill(0.0, length(speeds), trials, 2, n_points)

    # True CDF and parameters
    true_CDF = fill(0.0, length(speeds), n_points)
    CDF_plot_xs = fill(0.0, n_points)

    ###################################################
    # Compute estimates and bounds for CDF and params for different trials
    ###################################################

    for speed in speeds
        Threads.@threads for trial in 1:1:trials
            println(speed, ":", trial)

            fragment = data[speed+1, trial, :, :, :]

            # Guaranteed coverage bounds for CDF and other parameters
            # bound_xs, lb_ys, ub_ys = CDF_NS_band(fragment, props, δ)
            bound_xs, lb_ys, ub_ys = CDF_NS_band_optimized(fragment, props, δ)
            CDF_bounds[speed+1, trial, 1, :] = get_equispaced_CDF(bound_xs, lb_ys, props, n_points)[2]
            CDF_bounds[speed+1, trial, 2, :] = get_equispaced_CDF(bound_xs, ub_ys, props, n_points)[2]

        end;

        #######################
        # True CDF and parameters
        ########################
        xs, ys = on_policy_CDF(pi_data[speed+1, :])
        CDF_plot_xs, true_CDF[speed+1, :] = get_equispaced_CDF(xs, ys, props, n_points)
    end;


    ##############################################
    ## Save all the results from the experiments
    #################################################

    npzwrite(props["name"]*"_CDF_bounds.npy", CDF_bounds)
    npzwrite(props["name"]*"_true_CDF.npy", true_CDF)
    npzwrite(props["name"]*"_CDF_plot_xs.npy", CDF_plot_xs)


    #######################
    # Plots
    ########################

    function average_plot!(xs, ys, label)
        μs = dropdims(mean(ys, dims=1), dims=1)
        σs = dropdims(std(ys, dims=1), dims=1) ./ sqrt(trials)
        plot!(xs,μs, ribbon=σs, fillalpha=.3, label=label, legend=:left)
    end

    ########## CDF Plots ############

    plotCDF1 = plot(CDF_plot_xs, true_CDF[1, :], label="Fπ")                   # true CDF
    average_plot!(CDF_plot_xs, CDF_bounds[1, :, 1, :], "̂̂F-")             # lower bound CDF
    average_plot!(CDF_plot_xs, CDF_bounds[1, :, 2, :], "̂̂F+")             # upper bound CDF

    plotCDF2 = plot(CDF_plot_xs, true_CDF[2, :], label="Fπ")                   # true CDF
    average_plot!(CDF_plot_xs, CDF_bounds[2, :, 1, :], "̂̂F-")             # lower bound CDF
    average_plot!(CDF_plot_xs, CDF_bounds[2, :, 2, :], "̂̂F+")             # upper bound CDF

    plotCDF3 = plot(CDF_plot_xs, true_CDF[3, :], label="Fπ")                   # true CDF
    average_plot!(CDF_plot_xs, CDF_bounds[3, :, 1, :], "̂̂F-")             # lower bound CDF
    average_plot!(CDF_plot_xs, CDF_bounds[3, :, 2, :], "̂̂F+")             # upper bound CDF


    ###### Plot everything in a single plot ##########
    plot(plotCDF1, plotCDF2, plotCDF3, layout=(3, 1))

end





####
# For all of the domains, R \in [rmin, rmax] is always required
# If the range for return [Gmin, Gmax] is also known,
# then the code can use that to make bounds tighter as well.

domain_props = Dict(
    "Reco" => Dict(
                    "beta_path" => "../../Experiments/Reco/ActorCritic/DataDefault/0/Results/beta_trajectories_0.5.npy",
                    "pi_path" => "../../Experiments/Reco/ActorCritic/DataDefault/0/Results/eval_returns_0.5.npy",
                    "rmin" => 0.0,
                    "rmax" => 10.0,
                    "Gmin" => 0.0,
                    "Gmax" => 10.0,
                    "horizon" => 1,
                    "gamma" => 0.99,
                    "name" => "Reco"
    ),
    "NS_Reco" => Dict(
                    "beta_path" => "../../Experiments/NS_Reco/beta_trajectories_1000.npy",
                    "pi_path" => "../../Experiments/NS_Reco/eval_returns_1000.npy",
                    "rmin" => -1.0,
                    "rmax" => +1.0,
                    "Gmin" => -1.0,
                    "Gmax" => +1.0,
                    "horizon" => 1,
                    "gamma" => 1.0,
                    "name" => "NS_Reco"
    ),
    "Maze_POMDP" => Dict(
                    # POMDP + multiple behavior policies
                    "beta_path" => "../../Experiments/Maze/ActorCritic/DataDefault/0/Results/beta_trajectories_2.npy",
                    "pi_path" => "../../Experiments/Maze/ActorCritic/DataDefault/0/Results/eval_returns_0.75.npy",
                    "rmin" => -1.0,
                    "rmax" => +1.0,
                    "Gmin" => -6.0,
                    "Gmax" => -1.0,
                    "horizon" => 6,
                    "gamma" => 0.99,
                    "name" => "Maze_POMDP"
    ),
    "Diabetes" => Dict(
                    "beta_path" => "../../Experiments/SimGlucosediscrete-v0/ActorCritic/DataDefault/0/Results/beta_trajectories_0.5.npy",
                    "pi_path" => "../../Experiments/SimGlucosediscrete-v0/ActorCritic/DataDefault/0/Results/eval_returns_0.5.npy",
                    "rmin" => -15.0,
                    "rmax" => +15.0,
                    "Gmin" => -15.0,
                    "Gmax" => +15.0,
                    "horizon" => 1,
                    "gamma" => 0.99,
                    "name" => "Diabetes"
    )

)


function run_UnO(domain)
    props = domain_props[domain]
    beta_data = npzread(joinpath(@__DIR__, props["beta_path"]))
    pi_data = npzread(joinpath(@__DIR__, props["pi_path"]))
    compute_trend(props, beta_data, pi_data, 30, 2.0, 4.5, 6)
    # compute_trend(props, beta_data, pi_data, 3, 2.0, 4.0, 5)
end;


function run_NS_UnO(domain)
    props = domain_props[domain]
    beta_data = npzread(joinpath(@__DIR__, props["beta_path"]))
    pi_data = npzread(joinpath(@__DIR__, props["pi_path"]))
    compute_NS_trend(props, beta_data, pi_data, 30)
end;

#######################
# IMP: To make use threading
#
# Run in CLI before executing code
# export JULIA_NUM_THREADS=10
###########################

# run_UnO("Maze")
# run_UnO("Reco")
# run_UnO("Diabetes")

# run_UnO("Maze_POMDP")
# run_NS_UnO("NS_Reco")
