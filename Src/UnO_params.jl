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

include("./UnO_utils.jl")

#######################################################################
# Estimates for different parameters
#######################################################################??


function CDF_mean(xs, ys)
    """
    Plug-in mean estimate using a given CDF

    Args
    ----
    xs: x values
    ys: CDF estimates at those xs
    """

    ys_pdf = decumsum(ys)
    return sum(ys_pdf .* xs)
end


function CDF_variance(xs, ys)
    """
    Plug-in variance estimate using a given CDF

    Args
    ----
    xs: x values
    ys: CDF estimates at those xs
    """
    
    ys_pdf = decumsum(ys)
    temp_mean = sum(ys_pdf .* xs)
    return sum(ys_pdf .* ((xs .- temp_mean) .^2))
end


function CDF_entropy(xs, ys)
    """
    Plug-in mean entropy using a given CDF

    Args
    ----
    xs: x values
    ys: CDF estimates at those xs
    """
    ys_pdf = decumsum(ys)
    total = 0
    for item in ys_pdf
        if item > 0.00001
            total += item * log(item)
        end
    end
    return - total
    # return - sum(ys_pdf .* log.(ys_pdf))
end


function CDF_quantile(xs, ys, α)
    """
    Plug-in quantile estimate using a given CDF

    Args
    ----
    xs: x values
    ys: CDF estimates at those xs
    """
    id = searchsortedfirst(ys, α)
    id = min(id, length(ys))        # Overflow check because there might not be any ys > α
    return xs[id]
end


function CDF_IQR(xs, ys, α1, α2)
    """
    Plug-in inter-quantile range estimate using a given CDF

    Args
    ----
    xs: x values
    ys: CDF estimates at those xs
    """
    return CDF_quantile(xs, ys, α2) - CDF_quantile(xs, ys, α1)
end


function CDF_CVaR(xs, ys, α)
    """
    Plug-in conditional value-at-risk estimate using a given CDF

    Args
    ----
    xs: x values
    ys: CDF estimates at those xs
    """

    id = searchsortedfirst(ys, α)
    id = min(id, length(ys))     # Overflow check because there might not be any ys > α
    return (1/α) * CDF_mean(xs[1:id], clamp.(ys[1:id], 0, α))
end


#######################################################################
# CDF based bounds for different parameters
#######################################################################

function CDF_mean_bound(xs, lb_ys, ub_ys)
    """
    Bound for the mean

    Anderson (1967)
    https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/AND%20ONR%2001.pdf
    
    Args
    ----
    xs: Sorted sample values
    lb_ys: Lower bounds CDF at those xs
    ub_ys:  Upper bound CDF at those xs
    """

    lb_mean = CDF_mean(xs, ub_ys)  # Lower bound on mean is computed using upper bound of CDF
    ub_mean = CDF_mean(xs, lb_ys)  # Upper bound on mean is computed using lower bound of CDF
    return lb_mean, ub_mean
end


function CDF_variance_bound(xs, lb_ys, ub_ys)
    """
    Bound for the variance

    Romano et. al (2002)
    https://www.tandfonline.com/doi/abs/10.1081/STA-120006065?journalCode=lsta20

    Args
    ----
    xs: Sorted sample values
    lb_ys: Lower bounds CDF at those xs
    ub_ys:  Upper bound CDF at those xs
    """

    xs_n = length(xs)
    temp_xs = fill(0.0, xs_n + 1)
    temp_ys = fill(0.0, xs_n + 1)

    ###########################
    # Lower bound for variance
    ##########################
    function min_var!(xs, lb_ys, ub_ys, temp_xs, temp_ys, x_cutpoint)
        """
        This function creates a CDF that follows the lower bound CDF
        till the cutpoint, then does a vertical jump, 
        and then follows the upper bound CDF from there onwards
        """
        
        x_cutpoint = x_cutpoint[1]
        lower_id = searchsortedlast(xs, x_cutpoint)
        temp_xs[1:lower_id] = xs[1:lower_id]
        temp_xs[lower_id + 1] = x_cutpoint
        temp_xs[(lower_id + 2):(xs_n + 1)] = xs[(lower_id + 1):xs_n]

        temp_ys[1:lower_id] = lb_ys[1:lower_id]
        temp_ys[(lower_id + 1):(xs_n + 1)] = ub_ys[lower_id:xs_n]

        return CDF_variance(temp_xs, temp_ys)
    end

    res = bboptimize(candidate -> min_var!(xs, lb_ys, ub_ys, temp_xs, temp_ys, candidate),
                     SearchRange = (xs[1], last(xs)), NumDimensions = 1, TraceMode=:silent)
    var_lb = best_fitness(res)


    ###########################
    # Upper bound for variance
    ##########################
    function max_var!(xs, lb_ys, ub_ys, temp_xs, temp_ys, y_cutpoint)
        """
        This function creates a CDF that follows the upper bound CDF
        till the cutpoint, then does a horizontal jump, 
        and then follows the lower bound CDF from there onwards
        """
        
        y_cutpoint = y_cutpoint[1]
        upper_id = searchsortedfirst(ub_ys, y_cutpoint)
        upper_id2 = searchsortedfirst(lb_ys, y_cutpoint)
        temp_ys[1:(upper_id - 1)] = ub_ys[1:(upper_id - 1)]
        temp_ys[upper_id:(upper_id2 -1)] .= y_cutpoint
        temp_ys[upper_id2:(xs_n)] = lb_ys[upper_id2:xs_n]

        return - CDF_variance(xs[1:xs_n], temp_ys[1:xs_n]) # Negative because using max-opt but we need min
    end

    res = bboptimize(candidate -> max_var!(xs, lb_ys, ub_ys, temp_xs, temp_ys, candidate),
                     SearchRange = (0.0, 1.0), NumDimensions = 1, TraceMode=:silent)
    var_ub = - best_fitness(res)

    return var_lb, var_ub
end


function CDF_quantile_bound(xs, lb_ys, ub_ys, α)
    """
    Bound for the quantile

    Args
    ----
    xs: Sorted sample values
    lb_ys: Lower bounds CDF at those xs
    ub_ys:  Upper bound CDF at those xs
    α: Quantile level
    """
    lb_quantile = CDF_quantile(xs, ub_ys, α)   # Lower bound on quantile is computed using upper bound of CDF
    ub_quantile = CDF_quantile(xs, lb_ys, α)   # Upper bound on quantile is computed using lower bound of CDF
    return lb_quantile, ub_quantile
end


function CDF_IQR_bound(xs, lb_ys, ub_ys, α1, α2)
    """
    Bound for the inter-quantile range

    Args
    ----
    xs: Sorted sample values
    lb_ys: Lower bounds CDF at those xs
    ub_ys:  Upper bound CDF at those xs
    α1, α2: Quantile levels
    """

    # Lower bound on IQR: lowest for α2 - maximum for α1
    lb_IQR = CDF_quantile(xs, ub_ys, α2) - CDF_quantile(xs, lb_ys, α1)
    lb_IQR = max(0, lb_IQR)

    # Upper bound on IQR: maximum for α2 - minimum for α1
    ub_IQR = CDF_quantile(xs, lb_ys, α2) - CDF_quantile(xs, ub_ys, α1)

    return lb_VaR, ub_VaR
end


function CDF_CVaR_bound(xs, lb_ys, ub_ys, α)
    """
    Bound for the mean

    # Thomas et. al (2019)
    # https://people.cs.umass.edu/~pthomas/papers/Thomas2019.pdf
    
    Args
    ----
    xs: Sorted sample values
    lb_ys: Lower bounds CDF at those xs
    ub_ys:  Upper bound CDF at those xs
    """

    lb_CVaR = CDF_CVaR(xs, ub_ys, α)   # Lower bound on CVaR is computed using upper bound of CDF
    ub_CVaR = CDF_CVaR(xs, lb_ys, α)   # Upper bound on CVaR is computed using lower bound of CDF
    return lb_CVaR, ub_CVaR
end


function CDF_entropy_bound(xs, lb_ys, ub_ys)
    """
    Bound for the entropy 

    DeStefano et. al (2005)
    https://arxiv.org/ftp/cs/papers/0504/0504091.pdf

    The following black-box search method is naive and not ideal.
    Better way would be to actually code up the 'string-tightening'
    algorithm in the work by DeStefano et al.

    Args
    ----
    xs: Sorted sample values
    lb_ys: Lower bounds CDF at those xs
    ub_ys:  Upper bound CDF at those xs
    """

    return CDF_bbox_bound(xs, lb_ys, ub_ys, CDF_entropy)
end


function CDF_bbox_bound(xs, lb_ys, ub_ys, fn)
    """
    A generic blackbox optimization procedure to search for CDF within a band
    that optimizes for the desired parameter.
        
    The space of all CDFs is parameterized using "n_points". 
    Increasing n_points can lower the discretization error
    but can make the optimization problem harder.

    IMP: Typically, this should be the last resort. First shot should always be to leverage
    geometric insights as discussed in Figure 3 and 6 to obtain closed for solutions.
    """

    n_points = 50
    Gmin, Gmax = xs[1], last(xs)

    # Create constraints for the BBO search from the CDF band
    lb_constraints = get_equispaced_CDF2(xs, lb_ys, Gmin, Gmax, n_points)[2]
    ub_constraints = get_equispaced_CDF2(xs, ub_ys, Gmin, Gmax, n_points)[2]


    function wrapper(candidate, presign, fn)
        """
        Projects the candidate solution in the desired space (valid CDF + bounded)
        then computes the parameter on the projected CDF

        Args
        ----
        candidate: Candidate xs and ys 
        presign: Plus or negative to determine maximization or minimization
        fn: Plug-in estimator for the desired parameter
        """

        can_xs = candidate[1:n_points] .* abs(Gmax - Gmin)
        can_ys = candidate[n_points+1:2*n_points]

        candidate_xs = cumsum(can_xs)
        if sum(can_xs) > (Gmax - Gmin)
            candidate_xs = Gmin .+ (Gmax - Gmin) .* candidate_xs / sum(can_xs)
        else
            candidate_xs = Gmin .+ candidate_xs
        end

        lb_constraints = get_CDF_at_locs(xs, lb_ys, candidate_xs)
        ub_constraints = get_CDF_at_locs(xs, ub_ys, candidate_xs)
        candidate_ys = clamp.(cumsum(can_ys), lb_constraints, ub_constraints)

        return presign * fn(candidate_xs, candidate_ys)
    end

    # Ranges are set to (0, 0.1) as the values get rescaled in the wrapper
    
    # Lower bound for the parameter
    res = bboptimize(candidate -> wrapper(candidate, 1.0, fn),
                     SearchRange = (0.0, 0.1), NumDimensions = 2*n_points, TraceMode=:silent, method=:random_search)
    bbox_lb = best_fitness(res)

    # Upper bound for the parameter
    res = bboptimize(candidate -> wrapper(candidate, -1.0, fn), #negative fn
                     SearchRange = (0.0, 0.1), NumDimensions = 2*n_points, TraceMode=:silent, method=:random_search)
    bbox_ub = - best_fitness(res)

    return bbox_lb, bbox_ub
end


#######################################################################
# Bootstrap based bounds for different parameters
#######################################################################??

function CDF_mean_boot_bound(xs, ys, delta)
    return CDF_param_boot_bounds(xs, ys, delta, (x,y) -> CDF_mean(x,y))
end

function CDF_entropy_boot_bound(xs, ys, delta)
    return CDF_param_boot_bounds(xs, ys, delta, (x,y) -> CDF_entropy(x,y))
end


function CDF_variance_boot_bound(xs, ys, delta)
    return CDF_param_boot_bounds(xs, ys, delta, (x,y) -> CDF_variance(x,y))
end


function CDF_quantile_boot_bound(xs, ys, α, delta)
    return CDF_param_boot_bounds(xs, ys, delta, (x,y) -> CDF_quantile(x,y,α))
end


function CDF_IQR_boot_bound(xs, ys, α1, α2, delta)
    return CDF_param_boot_bounds(xs, ys, delta, (x,y) -> CDF_IQR(x,y,α1,α2))
end


function CDF_CVaR_boot_bound(xs, ys, α, delta)
    return CDF_param_boot_bounds(xs, ys, delta, (x,y) -> CDF_CVaR(x,y,α))
end


function CDF_param_boot_bounds(xs, ys, delta, param_fn)
    """
    Bootstrap based bounds for CDF's parameters
    
    Args
    ----
    xs: sorted sample values
    ys: CDF values at those xs
    delta:  Acceptable failure rate
    param_fn: Plug-in function for the desired parameter    
    """

    function get_param(x, y_pdf)
        """
        Helper function to re-sort data and compute the param estimate
        Resorting is required as re-sampling by bootstrap shuffles data.
        """
        order = sortperm(x)                 # Create order statistic map
        xs_boot = x[order]                  # Location of steps
        ys_boot = cumsum(y_pdf[order])      # Step heights
        return param_fn(xs_boot, ys_boot)
    end

    n = length(xs)
    idxs = collect(1:n)
    ys_pdf = decumsum(ys)

    bs = bootstrap(boot_id -> get_param(xs[boot_id], ys_pdf[boot_id]), idxs, BasicSampling(100))  

    try
        # Important:
        # In the bootstrap library, due to some unknown bug a variable zalpha becomes NaN
        # Another way to avoid error is by using the following change to the BCa code IN THE BOOTSTRAP LIBRARY:
        # zalpha == zalpha || return (t0, t0, t0)  # Yash: avoid the edge case of zalpha being NaN
        
        bci = confint(bs, BCaConfInt(1 - delta))                # BCa Bootstrap
        _, lb_param, ub_param = bci[1]                          # return format: estimate, lower, upper
        return lb_param, ub_param

    catch e
        println("BCa bootstrap error occured. Trying percentile bootstrap...")
        bci = confint(bs, PercentileConfInt(1 - delta))         # Percentile Bootstrap
        _, lb_param, ub_param = bci[1]                          # return format: estimate, lower, upper
        return lb_param, ub_param
    end

end
