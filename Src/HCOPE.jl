"""
This file contains the code for the work by 

P. S. Thomas, G. Theocharous, and M. Ghavamzadeh.
High Confidence Off-Policy Evaluation. 
In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.

Code repo: https://github.com/yashchandak/UnO
"""

using Statistics
using Random
using Bootstrap


function OPE_PDIS(data, props, ξr=0.0)
    """
    Off policy evaluation using per-decision importance sampling
     
    ----
    Args

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    ξr: can be used to shift the rewards such that it is always non-positive or non-negative 
    """

    # Extract the meta properties of the data
    n, horizon, d = size(data)
    γ = props["gamma"]

    total = fill(0.0, n)                        # Stores off-policy estimates for each trajectory
    for i in 1:1:n
        offreturn = 0.0                         # Initialize off-policy estimate for a trajectory
        ρ = 1.0                                 # Inititalize the importance ratio
        for j in 1:1:horizon
            ρ = ρ*data[i,j,1]                   # Product of past IS ratios
            reward = data[i,j,2] - ξr
            offreturn += γ^(j-1) * ρ * reward   # j-1 because first power should be 0
        end;
        total[i] = offreturn
    end;
    return total
end;


function OPE_IS(data, props, ξG=0.0)    
    """
    Off policy evaluation using full-trajectory importance sampling
     
    ----
    Args

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    ξG: can be used to shift the returns such that it is always non-positive or non-negative 
    """

    # Extract the meta properties of the data
    n, horizon, d = size(data)
    γ = props["gamma"]

    total = fill(0.0, n)
    for i in 1:1:n
        ρ = 1.0
        returns = - ξG
        for j in 1:1:horizon
            # Full trajectory IS
            ρ = ρ*data[i,j,1]
            returns += γ^(j-1) * data[i,j,2]
        end;
        total[i] = ρ * returns
    end;
    return total
end;


function CI_lower(data, props, δ=0.05)
    """
    Lower bound on the mean using concentration inequalities

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """

    # Extract meta-properties
    n, horizon, d = size(data)
    γ, T = props["gamma"], props["horizon"]
    rmin, gmin = props["rmin"], props["Gmin"]
    geom_sum = (1 - γ^T)/(1 - γ)

    # Get the importance sampled returns ** with control variate = rmin **
    estimates = OPE_PDIS(data, props, rmin)
    if any(x -> x<0, estimates)
        println("Bound will not work, items in array less than 0 (CI Lower)")
    end;

    #########################################################
    # Lower bound proposed by Thomas et al.  (2015)
    
    function lower_bound(D, c, n_post)
        Y = clamp.(D, 0.0, c)
        term1 = mean(Y)
        term2 = (7*c*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)

        lb = term1 - term2 - term3                      # Thm1 from Thomas et al. (2015)
        lb = lb + rmin * geom_sum                       # add back control to the bound (Thm 5 from Chandak et al. (2021))
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
    # Use D_pre to search for best truncation point 
    # using basic random search in 1D
    
    min_obs, max_obs = minimum(D_pre), maximum(D_pre)
    cs = (rand(100) .* (max_obs - min_obs)) .+ min_obs
    v_max, c_max = -Inf, cs[1]

    # Search for c that maximizes the lower bound
    for c in cs
        # Notice the use of n_post instead of n_pre for the size of data.
        # See the work by Thomas et al.  (2015) for discussion.
        val = lower_bound(D_pre, c, n_post)
        if val > v_max
            v_max, c_max = val, c
        end;
    end;

    ########################################################
    # Use D_post to obtain the final bound by Thomas et al. (2015)
    lb = lower_bound(D_post, c_max, n_post)

    # CI lower cannot be less than minimum possible returns
    lb = max(lb, geom_sum*rmin, gmin)

    return lb
end;



function CI_upper(data, props, δ=0.05)
    """
    Upper bound on the mean using concentration inequalities

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """

    # Extract meta-properties
    n, horizon, d = size(data)
    γ, T = props["gamma"], props["horizon"]
    rmax, gmax = props["rmax"], props["Gmax"]
    geom_sum = (1 - γ^T)/(1 - γ)

    # Get the importance sampled returns ** with control variate **
    estimates = OPE_PDIS(data, props, rmax)
    if any(x -> x>0, estimates)
        println("Bound will not work, items in array more than 0 (CI Upper)")
    end;


    #########################################################
    # Upper bound extension for the bound by Thomas et al.  (2015)
    # See Thm 8 in the work by Chandak et al. (2021) for details.

    function upper_bound(D, c, n_post)
        Y = clamp.(D, c, 0.0)
        term1 = mean(Y)
        term2 = (7*abs(c)*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)
        # println(term1, term2, term3)
        ub = term1 + term2 + term3                      # Thm1 from Thomas et al.  (2015)
        ub = ub + rmax*geom_sum                         # add back control to the bound (Thm 5 by Chandak et al. (2021))
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
    # Use D_pre to search for best truncation point 
    # using basic random search in 1D
    
    min_obs, max_obs = minimum(D_pre), maximum(D_pre)
    cs = (rand(100) .* (max_obs - min_obs)) .+ min_obs
    v_min, c_min = +Inf, cs[1]

    # Search for c that minimizes the upper bound
    for c in cs
        # Notice the use of n_post instead of n_pre for the size of data
        # See the work by Thomas et al.  (2015) for discussion.
        val = upper_bound(D_pre, c, n_post)
        if val < v_min
            v_min, c_min = val, c
        end;
    end;

    ########################################################
    # Use D_post to obtain the final bound 
    ub = upper_bound(D_post, c_min, n_post)

    # CI should not be higher than the maximum possible returns
    ub = min(ub, rmax*geom_sum, gmax)

    return ub
end;




function HCOPE_Bootstrap(data, props, δ=0.05)
    """
    Upper and Lower (approx) bound on the mean using Bootstrap

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """
    
    full_size, _, _ = size(data)

    IS_data = OPE_PDIS(data, props)
    bs = bootstrap(mean, IS_data, BasicSampling(100))   # Bootstrap mean's distribution
    # bci = confint(bs, BCaConfInt(0.95))               # Bootstrap CI on the mean
    bci = confint(bs, PercentileConfInt(1 - δ))         # Bootstrap CI on the mean

    estimate, lb, ub = bci[1]                           # Obtain the estiamte, lower bound, and the upper bound

    γ, T = props["gamma"], props["horizon"]
    rmin, gmin = props["rmin"], props["Gmin"]
    rmax, gmax = props["rmax"], props["Gmax"]
    geom_sum = (1 - γ^T)/(1 - γ)

    # Intersect bootstrap intervals with maximum and minimum possible return values
    lb = max(lb, gmin, geom_sum*rmin)
    ub = min(ub, gmax, geom_sum*rmax)

    return [estimate, lb, ub]
end;
