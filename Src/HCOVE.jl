"""
This file contains the code for the work by 

Y. Chandak, S. Shankar, and P. S. Thomas.
High Confidence Off-Policy (or Counterfactual) Variance Estimation. 
In Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence, 2021

Code repo: https://github.com/yashchandak/UnO
"""

using Statistics
using Random
using Bootstrap

include("./HCOPE.jl")


######################
# Variance Estimator
######################

function CDIS_fulltraj(data, props, ξG2=0.0)
    """
    Estimator for E[ρG^2] using full-traj IS

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    ξG2: Control variate to ensure G^2 is always non-positive or non-negative
    """

    # Meta-properties
    n, horizon, d = size(data)
    γ = props["gamma"]
    total = fill(0.0, n)

    for i in 1:1:n
        returns = 0.0
        ρ = 1.0
        for j in 1:1:horizon
            ρ = ρ * data[i,j,1]
            returns += γ^(j-1) * data[i,j,2]
        end;
        total[i] = ρ * (returns^2 - ξG2)
    end;
    return total
end;



function CDIS(data, props, ξr2=0.0)
    """
    Estimator for E[ρG^2] using 
    coupled decision importance sampling (CDIS) [Thm 3 by Chandak et al. (2021)]

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    ξr2: Control variate to ensure r^2 is always non-positive or non-negative
    """

    n, horizon, d = size(data)
    γ = props["gamma"]
    total = fill(0.0, n)

    for i in 1:1:n
        val = 0.0
        ρ_outer = 1.0

        ############
        # Alternate form
        # ∑_{j=0}^{T} ∑_{k=0}^T ρ(0,max(j, k)) γ^{j+k} R_j R_k
        # = ∑_{j=0}^T [ ρ(0, j)γ^{2j} R_j^2  + 2 ∑_{k=j+1}^T ρ(0,k) γ^{j+k} R_j R_k ]
        ############

        for j in 1:1:horizon
            ρ_outer = ρ_outer * data[i,j,1]
            R_j = data[i,j,2]
            val = val + ρ_outer * γ^(2*j-2) * (R_j^2 - ξr2)

            ρ_inner = ρ_outer
            for k in j+1:1:horizon
                ρ_inner = ρ_inner * data[i,k,1]
                R_k = data[i,k,2]
                val = val + 2 * ρ_inner * γ^(j+k-2) * (R_j * R_k - ξr2)
            end;
        end;
        total[i] = val
    end;
    return total
end;



function var_estimator_naive1(data, props)
    """
    This is the ̂σ^{!!} estimator 

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    """

    n, horizon, d = size(data)
    IS_returns = OPE_PDIS(data, props)

    return var(IS_returns)
end;


function var_estimator_naive2(data, props)
    """
    This is the ̂σ^{!} estimator 

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    """

    n, horizon, d = size(data)
    IS_returns = OPE_PDIS(data, props)

    μ = mean(IS_returns)

    T, γ = props["horizon"], props["gamma"]
    total = fill(0.0, n)
    for i in 1:1:n
        returns = 0.0
        ρ = 1.0
        for j in 1:1:horizon
            ρ = ρ * data[i,j,1]
            returns += γ^(j-1) * data[i,j,2]
        end;
        total[i] = ρ * (returns - μ)^2
    end;
    return sum(total)/(n-1)

end;


function var_estimator(data, props)
    """
    This is the ̂σ_n estimator 

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    """
    n, horizon, d = size(data)

    # Per/coupled decision variants
    term1 = mean(CDIS(data, props))                         # E[ρG^2]
    term2 = mean(OPE_PDIS(data[1:(n÷2), :, :], props))      # E[ρG] using first half of data
    term3 = mean(OPE_PDIS(data[(n÷2)+1:n, :, :], props))    # E[ρG] using second half of data

    return term1 - term2*term3                              # E[ρG^2] - E[ρG]E[ρG]
end;


function var_estimator_wo_CDIS(data, props)
    n, horizon, d = size(data)

    # No per-step CDIS
    term1 = mean(CDIS_fulltraj(data, props))
    term2 = mean(OPE_PDIS(data[1:(n÷2), :, :], props))
    term3 = mean(OPE_PDIS(data[(n÷2)+1:n, :, :], props))

    return term1 - term2*term3
end;




##########################################
#  High confidence Bounds
#######################################



function global_bounds(val, props)
    """
    Intersection of the proposed upper/lower values with
    lowest possible value and the higheest possible value (from Popoviciu's ineqality)

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    """

    rmin, rmax = props["rmin"], props["rmax"]
    gmin, gmax =  props["Gmin"], props["Gmax"]
    T, γ = props["horizon"], props["gamma"]

    geom_sum = (1 - γ^T)/(1 - γ)
    min_g = max(geom_sum*rmin, gmin)   # Maximum of the two lower possibilities
    max_g = min(geom_sum*rmax, gmax)   # Minimum of thw two upper possibilties

    max_var = ((max_g - min_g)^2) /4.0  # Maximum possible variance
    min_var = 0.0                       # Minimum possible variance

    return clamp.(val, min_var, max_var)
end;


function CDIS_lower(data, props, δ=0.05)
    """
    Lower bound on E[ρG^2]
    This uses the bound by Thomas et al. (2015) as a sub-routine

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """

    n, horizon, d = size(data)

    rmin, rmax = props["rmin"], props["rmax"]
    gmin, gmax =  props["Gmin"], props["Gmax"]
    T, γ = props["horizon"], props["gamma"]

    geom_sum = (1 - γ^T)/(1 - γ)
    min_g = max(geom_sum*rmin, gmin)       # Maximum of the two lower possibilities
    max_g = min(geom_sum*rmax, gmax)       # Minimum of thw two upper possibilties

    # minimum G^2 value possible
    min_g2 = 0.0
    if !(min_g < 0 && max_g > 0)
        # If the sign of min_g and max_g are the same
        min(min_g^2, max_g^2)
    end;

    # Get the CDIS returns
    # Note: ρG^2 is always positive
    # Therefore, no need to use control variates to subtract anything.
    estimates = CDIS(data, props, 0.0)

    if any(x -> x<0, estimates)
        println("CDIS Bound will not work, items in array less than 0")
    end;

    function lower_bound(D, c, n_post)
        Y = clamp.(D, 0.0, c)
        term1 = mean(Y)
        term2 = (7*c*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)

        lb = term1 - term2 - term3                      # Thm1 from Thomas (2015)
        # Note: No control variate shift is done here, as nothing was subtracted in CDIS

        return lb
    end;

    #########################################################
    # Prepare data splits into dpre and dpost
    # Size based on the split-value suggested by Thomas (2015)
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
        val = lower_bound(D_pre, c, n_post)
        if val > v_max
            v_max, c_max = val, c
        end;
    end;

    ########################################################
    # Bound by Thomas(2015)
    lb = lower_bound(D_post, c_max, n_post)

    # If lower bound is lower than minimum G^2 possible, then clip.
    lb = max(lb, min_g2)

    return lb
end;



function CDIS_upper(data, props, δ=0.05)
    """
    Upper bound on E[ρG^2]
    This uses the bound by Thomas et al. (2015) as a sub-routine

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """

    n, horizon, d = size(data)
    rmin, rmax = props["rmin"], props["rmax"]
    gmin, gmax =  props["Gmin"], props["Gmax"]
    T, γ = props["horizon"], props["gamma"]

    geom_sum = (1 - γ^T)/(1 - γ)
    ξr2 = max(rmin^2, rmax^2)               # maximum possible r^2 value

    min_g = max(geom_sum*rmin, gmin)        # Maximum of the two lower possibilities
    max_g = min(geom_sum*rmax, gmax)        # Minimum of thw two upper possibilties
    ξG2 = max(min_g^2, max_g^2)             # maximum G^2 value possible

    # Get the importance sampled returns ** with control variate **
    # Choose the control varaite smartly
    # if the full trajectory control is smaller, then use CDIS_fulltraj
    # Otherwise use CDIS with coupled decision control
    # Note: ξr2*(c^2) is never smaller than ξG2; it can only be equal to it at best.
    control = 0.0
    if ξG2 < ξr2*(geom_sum^2)
        estimates = CDIS_fulltraj(data, props, ξG2)
        control = ξG2
    else
        estimates = CDIS(data, props, ξr2)
        control = ξr2*(geom_sum^2)
    end;

    if any(x -> x>0, estimates)
        println("CDIS Bound will not work, items in array more than 0")
    end;

    function upper_bound(D, c, n_post)
        Y = clamp.(D, c, 0.0)
        term1 = mean(Y)
        term2 = (7*abs(c)*log(2/δ)) / (3*(n_post - 1))
        term3 = sqrt( (2 * var(Y) * log(2/δ) )/ n_post) # Sample variance form from Maurer(2009)

        ub = term1 + term2 + term3                      # Thm1 from Thomas (2015)
        ub = ub + control      # add back Gmax^2 to the bound
        return ub
    end;

    #########################################################
    # Prepare data splits into dpre and dpost
    # Size based on the split-value suggested by Thomas (2015)
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
        val = upper_bound(D_pre, c, n_post)
        if val < v_min
            v_min, c_min = val, c
        end;
    end;

    ########################################################
    # Bound by Thomas(2015)
    ub = upper_bound(D_post, c_min, n_post)

    # If upper bound is higher than maximum G^2 possible, then clip.
    ub = min(ub, ξG2)

    return ub
end;




function HCOVE_upper(data, props, δ=0.05)
    """
    Upper bound on the variance (Equation 6)

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """

    n, horizon, d = size(data)

    # δ/2 upper bound on E[ρG^2]
    term1 = CDIS_upper(data, props, δ/2)

    # δ/2 lower bound on E[ρG]^2
    # This requires two-sided bound on E[ρG]
    upper = CI_upper(data, props, δ/4)
    lower = CI_lower(data, props, δ/4)

    # Obtain lower bound on E[ρG]^2 from E[ρG] using bound propagation
    if lower <= 0.0 && upper >= 0.0
        term2 = 0.0
    else
        term2 = min(lower^2, upper^2)
    end;

    ub = term1 - term2

    # Bound by Popoviciu
    ub = global_bounds(ub, props)

    return ub
end;


function HCOVE_lower(data, props, δ=0.05)
    """
    Lower bound on the variance (Equation 5)

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate
    """
    
    n, horizon, d = size(data)

    # δ/2 lower bound on E[ρG^2]
    term1 = CDIS_lower(data, props, δ/2)

    # δ/2 upper bound on E[ρG]^2
    # This requires two-sided bound on E[ρG]
    upper = CI_upper(data, props, δ/4)
    lower = CI_lower(data, props, δ/4)

    # Obtain upper bound on E[ρG]^2 from E[ρG] using bound propagation
    term2 = max(lower^2, upper^2)

    lb = term1 - term2

    # Bound by Popoviciu
    lb = global_bounds(lb, props)

    return lb
end;


function HCOVE_Bootstrap(data, props)
    """
    Bootstrap based bounds for the variance

    Args
    ----

    data: Trajectories x Steps x (Importance ratio, Reward)
    prop: Meta properties of the data
    δ: Acceptable failure rate (0.05)
    """
    
    full_size, _, _ = size(data)
    idxs = collect(1:full_size)

    bs = bootstrap(x -> var_estimator(data[x, :, :], props), idxs, BasicSampling(100))   # Bootstrap mean's distribution
    # bci = confint(bs, BCaConfInt(0.95))               # Bootstrap CI on the mean
    bci = confint(bs, PercentileConfInt(0.95))          # Bootstrap CI on the mean

    estimate, lb, ub = bci[1]
    lb, ub = global_bounds([lb, ub], props)             # Intersect with the global possibility
    return [estimate, lb, ub]
end;
