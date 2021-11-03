"""
This file contains the code for the work by 

Y. Chandak, S. Niekum, B. Castro da Silva, E. Learned-Miller, E. Brunskill, and P. S. Thomas. 
Universal Off-Policy Evaluation. 
In Advances in Neural Information Processing Systems, 2021

Code repo: https://github.com/yashchandak/UnO
"""

using Statistics
using Random
using Bootstrap
using BlackBoxOptim

include("./UnO_params.jl")
include("./UnO_utils.jl")


#######################################################################
# CDF Estimators
#######################################################################


function IS(data, props, ξG=0.0)
    """
    Extract returns from behavior data
    and their associated IS weights wrt eval policy

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    ξG: Control variate to ensure G is always non-positive or non-negative
    """

    # Initialize arrays to store returns and IS weights
    n, horizon, d = size(data)
    γ = props["gamma"]
    returns = fill(0.0, n)
    ISweights = fill(0.0, n)

    for i in 1:1:n
        ρ = 1.0
        G = - ξG                        # Initialize with control-variate, if any
        for j in 1:1:horizon
            ρ = ρ*data[i,j,1]           # Full trajectory IS
            G += γ^(j-1) * data[i,j,2]
        end
        returns[i] = G
        ISweights[i] = ρ
    end

    return returns, ISweights
end


function IS_CDF(data, props)
    """
    Immportance sampling CDF

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    """

    returns, ISweights = IS(data, props)                # Get IS ratios
    order = sortperm(returns)                           # Create order statistic map

    xs = returns[order]                                 # Location of steps
    ys = cumsum(ISweights[order]) / length(ISweights)   # Step heights

    return xs, ys
end


function WIS_CDF(data, props)
    """
    Weighted Immportance sampling CDF

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    """

    returns, ISweights = IS(data, props)             # Get IS ratios
    order = sortperm(returns)                        # Create order statistics map

    xs = returns[order]                              # Location of steps
    ys = cumsum(ISweights[order]) / sum(ISweights)   # Step heights

    return xs, ys
end


#######################################################################
# Guarnateed Coverage Bounds
#######################################################################

function UnO_CI_lower(estimates, δ=0.05)
    """
    Construct a lower bound for CDF for a given keypoint
    This uses the bound by Thomas et al. (2015) as a sub-routine

    Args
    ----

    estimates: Sequence of samples for a keypoint κ: ρ_i 1{G_i ≤ κ}
    ξG: Control variate to ensure G is always non-positive or non-negative
    δ: Acceptable failure rate
    """

    n = length(estimates)
    if any(x -> x<-0.00001, estimates)
        println("Bound will not work, items in array less than 0 (UnO CI Lower)")

        for item in x
            if x < 0.0
                print(x, ",")
            end
        end
    end;

    function lower_bound(D, c, n_post)
        Y = clamp.(D, 0.0, c)
        term1 = mean(Y)
        term2 = (7*c*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)

        lb = term1 - term2 - term3                      # Thm1 from Thomas et al.  (2015)
        return lb
    end;

    #########################################################
    # Prepare data splits into dpre and dpost
    # Size based on the split-value suggested by Thomas et al.  (2015)
    n_pre = Int(floor(0.05 * n))
    n_post = n - n_pre

    perm_idxs = shuffle(collect(1:n))
    D_pre = estimates[perm_idxs[1:n_pre]]
    D_post = estimates[perm_idxs[n_pre + 1:n]]

    ##########################################################
    # Search for best truncation point using basic random search in 1D
    min_obs, max_obs = minimum(D_pre), maximum(D_pre)
    cs = (rand(100) .* (max_obs - min_obs)) .+ min_obs
    v_max, c_max = -Inf, cs[1]

    # Search for c that maximizes the lower bound
    for c in cs
        # Notice the use of n_post instead of n_pre.
        # See the work by Thomas et al.  (2015) for discussion.
        val = lower_bound(D_pre, c, n_post)
        if val > v_max
            v_max, c_max = val, c
        end;
    end;

    ########################################################
    # Bound by Thomas et al. (2015)
    lb = lower_bound(D_post, c_max, n_post)

    # Lower bound on a positive R.V. has should always be >= 0.
    lb = max(lb, 0)

    return lb
end;


function UnO_CI_upper(estimates, δ=0.05)   
    """
    Construct a lower bound for CDF for a given keypoint
    This uses the bound by Thomas et al. (2015) as a sub-routine

    Args
    ----

    estimates: Sequence of samples for a keypoint κ: ρ_i 1{G_i ≤ κ}
    ξG: Control variate to ensure G is always non-positive or non-negative
    δ: Acceptable failure rate
    """

    n = length(estimates)
    if any(x -> x>0.00001, estimates)
        println("Bound will not work, items in array more than 0 (UnO CI Upper)")
        for item in x
            if x > 0.0
                print(x, ",")
            end
        end
    end;

    function upper_bound(D, c, n_post)
        Y = clamp.(D, c, 0.0)
        term1 = mean(Y)
        term2 = (7*abs(c)*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)

        ub = term1 + term2 + term3                      # Thm1 from Thomas et al.  (2015)
        return ub
    end;

    #########################################################
    # Prepare data splits into dpre and dpost
    # Size based on the split-value suggested by Thomas et al.  (2015)
    n_pre = Int(floor(0.05 * n))
    n_post = n - n_pre

    perm_idxs = shuffle(collect(1:n))
    D_pre = estimates[perm_idxs[1:n_pre]]
    D_post = estimates[perm_idxs[n_pre + 1:n]]

    ##########################################################
    # Search for best truncation point using basic random search in 1D
    min_obs, max_obs = minimum(D_pre), maximum(D_pre)
    cs = (rand(100) .* (max_obs - min_obs)) .+ min_obs
    v_min, c_min = +Inf, cs[1]

    # Search for c that minimizes the upper bound
    for c in cs
        # Notice the use of n_post instead of n_pre.
        # See the work by Thomas et al.  (2015) for discussion.
        val = upper_bound(D_pre, c, n_post)
        # println(c, val)
        if val < v_min
            v_min, c_min = val, c
        end;
    end;

    ########################################################
    # Bound by Thomas et al. (2015)
    ub = upper_bound(D_post, c_min, n_post)

    # Upper bound on a negative R.V. should always be <= 0
    ub = min(ub, 0)

    return ub
end;


function CDF_band_with_KP_deltas(xs, ys, props, keypoints, deltas)
    """
    This function creates confidence intervals at the given keypoints
    Then under-completes and overcompletes it to get the entire confidence band

    Inputs:
    xs = x values for a CDF
    ys = y values for a CDF
    keypoints = location of the keypoints
    deltas = failure rate for each of the keypoints
    """

    n = length(xs)
    n_keys = length(keypoints)
    gmin, gmax = props["Gmin"], props["Gmax"]
    ys_pdf = decumsum(ys)

    temp = fill(0.0, n)
    # 2 extra positions for Gmin and Gmax
    bound_xs = fill(0.0, n_keys + 2)
    lb_ys = fill(0.0, n_keys + 2)       # Lower bounds
    ub_ys = fill(0.0, n_keys + 2)       # Upper bounds

    # Pre-specified bounds at the corner-points of CDF box
    bound_xs[1], bound_xs[n_keys+2] = gmin, gmax
    lb_ys[1] = 0
    ub_ys[n_keys+1] = 1
    lb_ys[n_keys+2] = 1
    ub_ys[n_keys+2] = 1

    for k in 1:1:n_keys
        key = keypoints[k]
        bound_xs[k+1] = key

        # Get lower bound at key
        idx = searchsortedlast(xs, key)
        temp[1:idx] = ys_pdf[1:idx] .* n        # Multiply by n to undo CDF normalization
        temp[idx+1:n] .= 0
        lb_ys[k+1] = UnO_CI_lower(temp, deltas[k]/2)

        # Get upper bound at key
        idx = searchsortedlast(xs, key)
        temp[1:idx] .= 0
        temp[idx+1:n] = - ys_pdf[idx+1:n] .* n      # Control variate form: ρ[1{G ≤ ν} - 1] = - ρ[1{G > ν}]

        ub_ys[k] = UnO_CI_upper(temp, deltas[k]/2) + 1  # Add back the expected control value
                                                    # Also, note ub_ys[k] instead of ub_ys[k+1]
                                                    # This left shift by 1 does overcompletion.
    end

    # Enforce monotonic consistency
    # Because CI used above has some randomness CI might not be monotonic
    # CI- for x2 > x1 can be enforced to be no lesser than the CI- for x1.
    # Similarly CI+ for all x1 < x2 can be enforced to be no greater than the CI+ for x2.
    for k in 1:1:n_keys
        lb_ys[k+1] = max(lb_ys[k], lb_ys[k+1])

        rev_i = n_keys - k + 1
        ub_ys[rev_i] = min(ub_ys[rev_i + 1], ub_ys[rev_i])
    end

    return bound_xs, lb_ys, ub_ys
end


function CDF_band(xs, ys, props, δ)
    """
    This function chooses equispaced keypoints where the CIs should be constructed.
    Then it creates CIs at those keypoints.
    It then under-completes and overcompletes it to get the confidence band.
    
    Inputs:
    xs = x values for a CDF
    ys = y values for a CDF
    keypoints = location of the keypoints
    deltas = failure rate for each of the keypoints
    """

    n_keys = ceil(Int32, log(length(xs)))
    # n_keys = ceil(Int32, sqrt(length(xs)))

    gmin, gmax = props["Gmin"], props["Gmax"]
    step = (gmax - gmin) / (n_keys + 1)

    keypoints = collect(gmin:step:gmax)[2:n_keys+1]     # Equi-spaced keypoints
    deltas = fill(δ/n_keys, n_keys)                 # Equal failure rates at each keypoint

    return CDF_band_with_KP_deltas(xs, ys, props, keypoints, deltas)
end


##################################################################
# Optimized CDF Bands with guaranteed coverage
##################################################################

function Optim_CI_lower(estimates, δ, n_post)
    """
    Construct a lower bound for CDF for a given keypoint
    This uses the bound by Thomas et al. (2015) as a sub-routine
        
        > This function is similar to CDF_CI_lower but the difference is 
          that this function is meant to be used during the trainig phase, and hence 
          does not split the data any further.

    Args
    ----

    estimates: Sequence of samples for a keypoint κ: ρ_i 1{G_i ≤ κ}
    δ: Acceptable failure rate
    """

    # Use all the data in the training mode
    # No further training and evaluation split

    if any(x -> x<-0.00001, estimates)
        println("Bound will not work, items in array less than 0 (Optim CI Lower)")
    end;

    function lower_bound(D, c, n_post)
        Y = clamp.(D, 0.0, c)
        term1 = mean(Y)
        term2 = (7*c*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)

        lb = term1 - term2 - term3                      # Thm1 from Thomas et al.  (2015)
        return lb
    end;

    ##########################################################
    # Search for best truncation point using basic random search in 1D
    min_obs, max_obs = minimum(estimates), maximum(estimates)
    cs = (rand(100) .* (max_obs - min_obs)) .+ min_obs
    v_max, c_max = -Inf, cs[1]

    # Search for c that maximizes the lower bound
    for c in cs
        # Notice the use of n_post instead of n_pre.
        # See the work by Thomas et al.  (2015) for discussion.
        val = lower_bound(estimates, c, n_post)
        if val > v_max
            v_max, c_max = val, c
        end;
    end;

    # Lower bound on a positive R.V. should always be >= 0.
    lb = max(v_max, 0)

    return lb
end;


function Optim_CI_upper(estimates, δ, n_post)
    """
    Construct an upper bound for CDF for a given keypoint
    This uses the bound by Thomas et al. (2015) as a sub-routine
        
        > This function is similar to CDF_CI_lower but the difference is 
          that this function is meant to be used during the trainig phase, and hence 
          does not split the data any further.

    Args
    ----

    estimates: Sequence of samples for a keypoint κ: ρ_i 1{G_i ≤ κ}
    δ: Acceptable failure rate
    """

    if any(x -> x>0.00001, estimates)
        println("Bound will not work, items in array more than 0 (Optim CI Upper)")
    end;

    function upper_bound(D, c, n_post)
        Y = clamp.(D, c, 0.0)
        term1 = mean(Y)
        term2 = (7*abs(c)*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)

        ub = term1 + term2 + term3                      # Thm1 from Thomas et al.  (2015)
        return ub
    end;

    ##########################################################
    # Search for best truncation point using basic random search in 1D
    min_obs, max_obs = minimum(estimates), maximum(estimates)
    cs = (rand(100) .* (max_obs - min_obs)) .+ min_obs
    v_min, c_min = +Inf, cs[1]

    # Search for c that minimizes the upper bound
    for c in cs
        # Notice the use of n_post instead of n_pre.
        # See the work by Thomas et al.  (2015) for discussion.
        val = upper_bound(estimates, c, n_post)
        # println(c, val)
        if val < v_min
            v_min, c_min = val, c
        end;
    end;

    # Upper bound on a negative R.V. should always be <= 0
    ub = min(v_min, 0)

    return ub
end;


function Optim_band_with_KP_deltas(xs, ys, props, keypoints, deltas, n_post)
    """
    This function creates confidence intervals at the given keypoints
    Then under-completes and overcompletes it to get the entire confidence band
    
        > This function is similar to CDF_band_with_KP_deltas but the difference is 
          that this function is meant to be used during the trainig phase, and hence 
          does not split the data any further for creating the upper and lower CIs

    Inputs:
    xs = x values for a CDF
    ys = y values for a CDF
    keypoints = location of the keypoints
    deltas = failure rate for each of the keypoints
    """

    # Obtain meta-properties
    n = length(xs)
    n_keys = length(keypoints)
    gmin, gmax = props["Gmin"], props["Gmax"]
    ys_pdf = decumsum(ys)

    # Initialize the bound variables
    temp = fill(0.0, n)
    bound_xs = fill(0.0, n_keys + 2)    # 2 extra positions for Gmin and Gmax
    lb_ys = fill(0.0, n_keys + 2)       # Lower bounds
    ub_ys = fill(0.0, n_keys + 2)       # Upper bounds

    # Pre-specified bounds at the corner-points of CDF box
    bound_xs[1], bound_xs[n_keys+2] = gmin, gmax
    lb_ys[1] = 0
    ub_ys[n_keys+1] = 1
    lb_ys[n_keys+2] = 1
    ub_ys[n_keys+2] = 1

    # FOr each keypoint get the upper and lower bounds
    for k in 1:1:n_keys
        key = keypoints[k]
        bound_xs[k+1] = key

        # Get lower bound at key
        idx = searchsortedlast(xs, key)
        # Estimates using samples less than key = ρ_i G_i, and estimates after key = 0
        temp[1:idx] = ys_pdf[1:idx] .* n        # Multiply by n to undo CDF normalization
        temp[idx+1:n] .= 0
        lb_ys[k+1] = Optim_CI_lower(temp, deltas[k]/2, n_post)

        # Get upper bound at key
        idx = searchsortedlast(xs, key)
        temp[1:idx] .= 0
        temp[idx+1:n] = - ys_pdf[idx+1:n] .* n      # Control variate form: ρ[1{G ≤ ν} - 1] = - ρ[1{G > ν}]
        ub_ys[k] = Optim_CI_upper(temp, deltas[k]/2, n_post) + 1    # Add back the expected control value
                                                                    # Also, note ub_ys[k] instead of ub_ys[k+1]
                                                                    # This left shift by 1 does overcompletion.
    end

    # Enforce monotonic consistency
    # Because CI used above has some randomness CI might not be monotonic
    # CI- for x2 > x1 can be enforced to be no lesser than the CI- for x1.
    # Similarly CI+ for all x1 < x2 can be enforced to be no greater than the CI+ for x2.
    for k in 1:1:n_keys
        lb_ys[k+1] = max(lb_ys[k], lb_ys[k+1])

        rev_i = n_keys - k + 1
        ub_ys[rev_i] = min(ub_ys[rev_i + 1], ub_ys[rev_i])
    end

    return bound_xs, lb_ys, ub_ys
end


function CDF_band_optimized(xs, ys, props, δ)
    """
    This function optmizes for the keypoints where the CIs should be constructed.
    Then it creates CIs at those keypoints.
    It then under-completes and overcompletes it to get the confidence band.
    
    IMPORTANT: If bounds for only a single parameter is required, 
    then the following optimization procedure can be improved. Instead of optimizing for the area enlcosed,
    one can directly optimize for the desired parameter by changing the area_fn() below.
    See Appendix E.3 in the work by Chandak et al. (2021) for more details.

    Inputs:
    xs = x values for a CDF
    ys = y values for a CDF
    keypoints = location of the keypoints
    deltas = failure rate for each of the keypoints
    """

    # Obtain meta-properties
    n_keys = ceil(Int32, log(length(xs)))       # Number of keypoints to be used (hyper-parameter)
    # n_keys = ceil(Int32, sqrt(length(xs)))
    gmin, gmax = props["Gmin"], props["Gmax"]
    step = (gmax - gmin) / (n_keys + 1)

    # Split data into train and test set
    # Train set is used for doing all the optimization
    # Test set is used to get the actual CDF band
    n = length(xs)
    n_train = Int(floor(0.05 * n))
    n_post = n - n_train

    ys_unnorm_pdf = decumsum(ys) .* n       # Basically get back IS ratios

    perm_idxs = shuffle(collect(1:n))
    x_train = xs[perm_idxs[1:n_train]]
    y_train = ys_unnorm_pdf[perm_idxs[1:n_train]]

    x_post = xs[perm_idxs[n_train+1:n]]
    y_post = ys_unnorm_pdf[perm_idxs[n_train+1:n]]

    ##################
    # Training phase
    ##################

    order = sortperm(x_train)
    xs_train = x_train[order]
    ys_train = cumsum(y_train[order]) / n_train

    function constraints(candidate)
        """
        Helper function to project the candidate solution for location and the delta 
        at each keypoint within the desired limits. 

        (This is one choice. There could be other ways to enforce these constraints 
        that make the optimization procedure better in some sense)
        """
        key_locs = candidate[1:length(candidate)÷2]
        key_dels = candidate[(length(candidate)÷2+1):length(candidate)]

        # Rescale individual leypoint locs to ensure they cover the desired range
        temp1 = cumsum(key_locs)
        if sum(key_locs) > (gmax - gmin)
            temp1 = gmin .+ (gmax - gmin) .* temp1 / sum(key_locs)
        else
            temp1 =  gmin .+ temp1
        end

        # Ensure sum of individual deltas is never more than δ 
        temp2 = cumsum(key_dels)
        if sum(key_dels) > δ
            temp2 = δ .* temp2 / sum(temp2)
        end

        return temp1, temp2
    end

    function area_fn(candidate)
        """
        Helper function to compute the total area enclosed within the CDF band
        """

        # As this is training phase, use `optim' version of KP delta which does not create further data splits.
        candidate_keys, candidate_deltas = constraints(candidate)
        bound_xs, lb_ys, ub_ys = Optim_band_with_KP_deltas(xs_train, ys_train, props,
                                                           candidate_keys, candidate_deltas, n_post)

        # Compute the area within the CDF band
        area = 0
        for i in 1:1:length(bound_xs) - 1
            height = ub_ys[i] - lb_ys[i]
            width = bound_xs[i+1] - bound_xs[i]
            area = area + height * width
        end

        return area
    end

    # Use blackbox search to find a good set of keypoint locs and deltas
    # For blackbox search, define the range of domain
    # (Notice, through constrain projection, locs get rescaled to: gmin to gmax) 
    lb_range = fill(0.0, n_keys*2)
    ub_range = fill(δ, n_keys*2)
    ub_range[1:n_keys] .= abs(gmax-gmin)
    res = bboptimize(area_fn, SearchRange = collect(zip(lb_range, ub_range)), NumDimensions = n_keys*2,
                                            method=:random_search, TraceMode=:silent)

    # Get the optimal key points
    best_keypoints, best_deltas = constraints(best_candidate(res))

    ########################
    # Final evaluation phase
    #########################

    order = sortperm(x_post)
    xs_post = x_post[order]
    ys_post = cumsum(y_post[order]) / n_post

    # For final eval, use non-optim version of KP_delta function such that multi-comparison is avoided
    return CDF_band_with_KP_deltas(xs_post, ys_post, props, best_keypoints, best_deltas)
end



#######################################################################
# Approximate Bounds Using Statistical Bootstrap
#######################################################################

function CDF_boot_band(xs, ys, props, delta)
    """    
    This was a preliminary attempt at developing bootstrap based Off-policy CDF bounds.

    We tried a functional bootstrap based approach below, but it does not work well.
    Avoid using it. Instead, one can easily bound any parameter directly when using bootstrap, 
    instead of bounding CDF as an intermediate step (which was helpful when using concentration inequalities)
    """

    function get_distance(x, y_pdf)
        order = sortperm(x)                 # Create order statistic map
        xs_boot = x[order]                  # Location of steps
        ys_boot = cumsum(y_pdf[order])      # Step heights
        return KS_distance(xs, ys, xs_boot, ys_boot)
    end

    n = length(xs)
    idxs = collect(1:n)
    ys_pdf = decumsum(ys)

    bs = bootstrap(boot_id -> get_distance(xs[boot_id], ys_pdf[boot_id]), idxs, BasicSampling(100))   # Bootstrap KS distance's distribution
    # bci = confint(bs, BCaConfInt(1 - 2*delta))            # BCa Bootstrap CI on the KS distance
    bci = confint(bs, PercentileConfInt(1 - 2*delta))       # Percentile Bootstrap CI on the KS distance
                                                            # Compute with 2*delta because it internally divides
                                                            # by 2 to get bounds for both the tails
                                                            # However, we only need the bound for the upper tail

    _, _, ub_dist = bci[1]      # return format: estimate, lower, upper

    # 2 extra positions for Gmin and Gmax
    gmin, gmax = props["Gmin"], props["Gmax"]
    bound_xs = fill(0.0, n+2)
    lb_ys = fill(0.0, n+2)       # Lower bounds
    ub_ys = fill(0.0, n+2)       # Upper bounds

    # Pre-specified bounds at the corner-points of CDF box
    bound_xs[1], bound_xs[n+2] = gmin, gmax
    bound_xs[2:n+1] = xs
    lb_ys[1] = 0
    ub_ys[n+1] = 1
    lb_ys[n+2] = 1
    ub_ys[n+2] = 1

    # Create the upper (over-complete) and lower (under-complete) CDF
    lb_ys[2:n+1] = ys .- ub_dist
    ub_ys[1:n] = ys .+ ub_dist      # Note ub_ys[1:n] instead of ub_ys[2:n+1]
                                    # This left shift by 1 does overcompletion.

    # Bounds can be clipped between 0 and 1
    clamp!(lb_ys, 0, 1)
    clamp!(ub_ys, 0, 1)

    return bound_xs, lb_ys, ub_ys
end


#######################################################################
# Approximate Bounds for Non-stationary Setting Using Wild Bootstrap
#######################################################################

function CDF_NS_band_with_KP_deltas(eps, returns, ISweights, props, keypoints, deltas, L, mode)
    """
    This function creates wild-bootstrap based confidence intervals at the given keypoints
    Then under-completes and overcompletes it to get the entire confidence band

    Inputs:
    xs = x values for a CDF
    ys = y values for a CDF
    keypoints = location of the keypoints
    deltas = failure rate for each of the keypoints
    """

    n = length(returns)
    n_keys = length(keypoints)
    gmin, gmax = props["Gmin"], props["Gmax"]

    temp = fill(0.0, n)
    # 2 extra positions for Gmin and Gmax
    bound_xs = fill(0.0, n_keys + 2)
    lb_ys = fill(0.0, n_keys + 2)       # Lower bounds
    ub_ys = fill(0.0, n_keys + 2)       # Upper bounds

    # Pre-specified bounds at the corner-points of CDF box
    bound_xs[1], bound_xs[n_keys+2] = gmin, gmax
    lb_ys[1] = 0
    ub_ys[n_keys+1] = 1
    lb_ys[n_keys+2] = 1
    ub_ys[n_keys+2] = 1

    for k in 1:1:n_keys
        key = keypoints[k]
        bound_xs[k+1] = key

        # Get lower bound at key
        idx = returns .<= key
        temp = ISweights .* idx
        lb_ys[k+1], ub_ys[k] = nswildbs_CI(eps, temp, deltas[k], L, mode)
        # lb_ys[k+1], ub_ys[k] = nswildbst_CI(eps, temp, deltas[k], n)
    end

    # Enforce monotonic consistency
    # Because CI used above has some randomness CI might not be monotonic
    # CI- for x2 > x1 can be enforced to be no lesser than the CI- for x1.
    # Similarly CI+ for all x1 < x2 can be enforced to be no greater than the CI+ for x2.
    for k in 1:1:n_keys
        lb_ys[k+1] = min(1, max(lb_ys[k], lb_ys[k+1]))

        rev_i = n_keys - k + 1
        ub_ys[rev_i] = max(0, min(ub_ys[rev_i + 1], ub_ys[rev_i]))
    end

    return bound_xs, lb_ys, ub_ys
end


function CDF_NS_band(data, props, δ)
    """
    This function constucts equi-spaced keypoints and then creates wild-bootstrap based bounds at those keypoints.
    It then under-completes and overcompletes it to get the confidence band.

        > This function is similar to CDF_band() but used wild-bootstrap instead of CIs
    
    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """

    # Immportance sampling CDF
    returns, ISweights = IS(data, props)                # Get IS ratios

    n_keys = ceil(Int32, log(length(returns)))
    # n_keys = ceil(Int32, sqrt(length(returns)))

    gmin, gmax = props["Gmin"], props["Gmax"]
    step = (gmax - gmin) / (n_keys + 1)

    keypoints = collect(gmin:step:gmax)[2:n_keys+1]     # Equi-spaced keypoints
    deltas = fill(δ/n_keys, n_keys)                 # Equal failure rates at each keypoint

    eps = collect(Float64, 1:length(returns))
    return CDF_NS_band_with_KP_deltas(eps, returns, ISweights, props, keypoints, deltas, last(eps), :normal)
end


function CDF_NS_band_optimized(data, props, δ)
    """
    This function optmizes for the keypoints where the CIs should be constructed.
    Then it creates wild-bootstrap based bounds at those keypoints.
    It then under-completes and overcompletes it to get the confidence band.
    
    IMPORTANT: If bounds for only a single parameter is required, 
    then the following optimization procedure can be improved. Instead of optimizing for the area enlcosed,
    one can directly optimize for the desired parameter by changing the area_fn() below.
    See Appendix E.3 in the work by Chandak et al. (2021) for more details.
    
    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """

    # Importance sampling CDF
    returns, ISweights = IS(data, props)                # Get IS ratios
    n_keys = ceil(Int32, log(length(returns)))
    # n_keys = ceil(Int32, sqrt(length(returns)))


    # Obtain meta-properties
    gmin, gmax = props["Gmin"], props["Gmax"]
    step = (gmax - gmin) / (n_keys + 1)

    # Variables for data splitting
    n = length(returns)
    n_train = Int(floor(0.5 * n))
    n_post = n - n_train
    perm_idxs = shuffle(collect(1:n))


    #############################
    # Training phase
    # Similar to CDF_band_optimized, 
    # but uses wild bootstrap instead of concentration inequalities
    #############################

    eps_train = perm_idxs[1:n_train]
    return_train = returns[eps_train]
    ISweights_train = ISweights[eps_train]

    function constraints(candidate)
        """
        Projects candidate solutions to the feasible/desired set.
        """

        key_locs = candidate[1:length(candidate)÷2]
        key_dels = candidate[(length(candidate)÷2+1):length(candidate)]

        temp1 = cumsum(key_locs)
        if sum(key_locs) > (gmax - gmin)
            temp1 = gmin .+ (gmax - gmin) .* temp1 / sum(key_locs)
        else
            temp1 =  gmin .+ temp1
        end

        temp2 = cumsum(key_dels)
        if sum(key_dels) > δ
            temp2 = δ .* temp2 / sum(temp2)
        end

        return temp1, temp2
    end

    function area_fn(candidate)
        """
        Area of the CDF band
        """

        candidate_keys, candidate_deltas = constraints(candidate)
        bound_xs, lb_ys, ub_ys = CDF_NS_band_with_KP_deltas(eps_train, return_train, ISweights_train, props, candidate_keys, candidate_deltas, n, :fast)

        area = 0
        for i in 1:1:length(bound_xs) - 1
            height = ub_ys[i] - lb_ys[i]
            width = bound_xs[i+1] - bound_xs[i]
            area = area + height * width
        end

        return area
    end

    lb_range = fill(0.0, n_keys*2)
    ub_range = fill(δ, n_keys*2)
    ub_range[1:n_keys] .= abs(gmax-gmin)
    res = bboptimize(area_fn, SearchRange = collect(zip(lb_range, ub_range)), NumDimensions = n_keys*2,
                                            method=:random_search, TraceMode=:silent)

    # Get the optimal key points
    best_keypoints, best_deltas = constraints(best_candidate(res))

    ##############################
    # Final evaluation phase
    ##############################

    eps_post = perm_idxs[n_train+1:n]
    return_post = returns[eps_post]
    ISweights_post = ISweights[eps_post]

    return CDF_NS_band_with_KP_deltas(eps_post, return_post, ISweights_post, props, best_keypoints, best_deltas, n, :normal)
end
