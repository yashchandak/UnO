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
using LinearAlgebra

#######################################################################
# General Utilities
#######################################################################??

function decumsum(xs::Array{Float64, 1})
    """
    Create empirical probability distribution from empirical CDF
    """

    N = length(xs)
    temp = fill(0.0, N)
    temp[1] = xs[1]

    # First element stays as-is
    for i in 2:1:N
        temp[i] = xs[i] - xs[i-1]
    end

    return temp
end


function KS_distance(xs1, ys1, xs2, ys2)
    """
    Kolmogorov-Smirnov distance between 2 CDFs
    
    Inputs
    xs1, ys1, for the first CDF
    xs2, ys2, for the second CDF
    """

    function dist(x1, y1, x2, y2)
        # Check the distance between CDF1 and CDF2, 
        # only at the points in CDF1

        n = length(x1)
        n2 = length(x2)
        j = 1
        max_d = -Inf
        for i in 1:1:n
            if j <= n2
                while x2[j] <= x1[i]
                    j = j+1
                    if j > n2; break; end # Prevent overflowing
                end
            end

            # Above while loop stops immediately when x2[j] crosses x1[i]
            # Therefore decrement j by one to get 2nd cdf value at x1[i]
            if j==1
                max_d = max(max_d, abs(y1[i] - 0))
            else
                max_d = max(max_d, abs(y1[i] - y2[j-1]))
            end
        end
        return max_d
    end

    # Search both ways: CDF1 vs CDF2, and CDF2 vs CDF1
    d1 = dist(xs1, ys1, xs2, ys2)
    d2 = dist(xs2, ys2, xs1, ys1)
    return max(d1, d2)
end


function get_equispaced_CDF(xs, ys, props, n_points=100)
    """
    Create a CDF at equispaced points from the sample CDF
    This function is particularly useful for plotting
    and averaging over multiple trials

    Note: This is only an approximation because of discretization
    """

    Gmin, Gmax = props["Gmin"], props["Gmax"]
    steps = (Gmax - Gmin)/(n_points - 1)

    # Initialize equi-spaced CDF
    Y_CDF = fill(0.0, n_points)
    X_points = fill(0.0, n_points)
    n = length(xs)

    j, k = 1, 1                     # Indexing starts at 1 in Julia
    for i in Gmin:steps:Gmax
        X_points[j] = i

        # Search for the smallest observed return greater than i
        if k <= n
            while xs[k] <= X_points[j]
                k = k+1
                if k > n; break; end # handle overflow
            end
        end

        if k == 1
            # If i is less than smallest observed return then set CDF = 0
            Y_CDF[j] = 0
        else
            # Otherwise CDF at i is the
            # estimated CDF of the largest observed return smaller than i
            Y_CDF[j] = ys[k-1]
        end

        j = j + 1
    end

    return X_points, Y_CDF
end


function get_equispaced_CDF2(xs, ys, Gmin, Gmax, n_points=100)
    """
    Create a CDF at equispaced points from the sample CDF
    This function is particularly useful for plotting
    and averaging over multiple trials
    
    Note: This is only an approximation because of discretization
    """

    steps = (Gmax - Gmin)/(n_points - 1)

    # Initialize equi-spaced CDF
    Y_CDF = fill(0.0, n_points)
    X_points = fill(0.0, n_points)
    n = length(xs)

    j, k = 1, 1                     # Indexing starts at 1 in Julia
    for i in Gmin:steps:Gmax
        X_points[j] = i

        # Search for the smallest observed return greater than i
        if k <= n
            while xs[k] <= X_points[j]
                k = k+1
                if k > n; break; end # handle overflow
            end
        end

        if k == 1
            # If i is less than smallest observed return then set CDF = 0
            Y_CDF[j] = 0
        else
            # Otherwise CDF at i is the
            # estimated CDF of the largest observed return smaller than i
            Y_CDF[j] = ys[k-1]
        end

        j = j + 1
    end

    return X_points, Y_CDF
end

function get_CDF_at_locs(xs, ys, X_points)
    """
    Get CDF values at some alternate points: X_points
    """

    n_points = length(X_points)
    Y_CDF = fill(0.0, n_points)
    n = length(xs)

    j, k = 1, 1                     # Indexing starts at 1 in Julia
    for i in X_points

        # Search for the smallest observed return greater than i
        if k <= n
            while xs[k] <= i
                k = k+1
                if k > n; break; end # handle overflow
            end
        end

        if k == 1
            # If i is less than smallest observed return then set CDF = 0
            Y_CDF[j] = 0
        else
            # Otherwise CDF at i is the
            # estimated CDF of the largest observed return smaller than i
            Y_CDF[j] = ys[k-1]
        end

        j = j + 1
    end

    return Y_CDF
end




function on_policy_CDF(pi_data)
    """
    Empirical CDF computed using on-policy Monte-carlo
    """

    N = length(pi_data)
    xs = sort(pi_data);
    ys =  collect(1.0/N:1.0/N:1.0)

    return xs, ys
end


function behavior_CDF(data, props)
    """
    Empirical CDF for the behavior policy
    """

    n, horizon, d = size(data)
    γ = props["gamma"]
    returns = fill(0.0, n)

    # Extract returns under behavior policy
    for i in 1:1:n
        G = 0
        for j in 1:1:horizon
            G += γ^(j-1) * data[i,j,2]
        end
        returns[i] = G
    end

    return on_policy_CDF(returns)
end



################################################
# Wild bootstrap stuff
# Percentile-based
#
# Y. Chandak, S. Jordan, G. Theocharous, M. White, and P. S. Thomas.
# Towards Safe Policy Improvement for Non-Stationary MDPs.
# In Advances in Neural Information Processing Systems, 2020.
#############################################


function get_coefs(Φ, ϕτ)
    """
    get_coefs(Φ, ϕτ)
    returns least squares coefficients for making predictions
    at observed points Φ and points of interest ϕτ.
    This function is a helper function to be used for predicting
    future points in a time series and for in wild bootstrap.
    """
    H = pinv(Φ' * Φ) * Φ'
    W = Φ * H
    ϕ = ϕτ * H
    return W, ϕ
end


function get_preds_and_residual(Y, W, ϕ)
    """
    get_preds_and_residual(Y, W, ϕ)
    returns the baseline predictions using observed labels Y at points ϕ
    and along with the vector of residuals Y - Ŷ.
    W and ϕ should be the output of get_coefs.
    """
    Ŷ = W*Y
    y = ϕ*Y
    ξ = Y .- Ŷ
    return y, ξ
end


function wildbs_eval(y, ξ, ϕ, σ, f)
        """
        wildbs_eval(y, ξ, ϕ, σ, f)
    evaluates the prediction of a linear timeseries model using
    noise generated for the wild bootstrap.
    y is the prediction at points ϕ using original labels Y
    ξ are the residual between the labels Y and prediction Ŷ
    ϕ are the points (coefficients) at at which the predictions y are made
    σ is the noise sample used to modify the residual, e.g., vector of {-1, 1}
    f is the function to aggregate the result for all points in ϕ, e.g., sum, mean, maximum, minimum
    """
    return f(y .+ ϕ * (ξ .* σ))
end


function wildbs_CI(y, ξ, ϕ, δ, num_boot, aggf, rng, mode=:normal)
    """
    wildbs_CI(y, ξ, ϕ, δ, num_boot, aggf)
    computes the δ percentiles bootstrap using the wild bootstrap method with num_boot bootstrap samples
    for original predictions y, with residuals ξ, at features ϕ, aggregated by aggf.
    """
    # bsamples = [wildbs_eval(y, ξ, ϕ, sign.(randn(rng, length(ξ))), aggf) for i in 1:num_boot]
    # bsamples = [wildbs_eval(y, ξ, ϕ, sign.(randn(length(ξ))), aggf) for i in 1:num_boot]        # Without random seed
    # return quantile(sort(bsamples), δ, sorted=true)
    function get_val()
        return wildbs_eval(y, ξ, ϕ, sign.(randn(length(ξ))), aggf)
    end

    n = length(ξ)
    idxs = collect(1:n)

    if mode == :fast
        bs = bootstrap(boot_id -> get_val(), idxs, BasicSampling(100))
        bci = confint(bs, PercentileConfInt(1 - 2*δ[1]))
        _, lb_param, ub_param = bci[1]
        return lb_param, ub_param
    end

    ##########
    # Use BCa
    ###############

    bs = bootstrap(boot_id -> get_val(), idxs, BasicSampling(num_boot))   # Bootstrap KS distance's distribution

    try
        # Important:
        # In the bootstrap library, due to some unknown bug a variable zalpha becomes NaN
        # Another way to avoid error is by using the following change to the BCa code IN THE BOOTSTRAP LIBRARY:
        # zalpha == zalpha || return (t0, t0, t0)  # Yash: avoid the edge case of zalpha being NaN
        bci = confint(bs, BCaConfInt(1 - 2*δ[1]))              # BCa Bootstrap CI 
        _, lb_param, ub_param = bci[1]      # return format: estimate, lower, upper
        return lb_param, ub_param
    catch e
        println("BCa bootstrap error occured. Trying percentile bootstrap...")
        bci = confint(bs, PercentileConfInt(1 - 2*δ[1]))       # Percentile Bootstrap CI 
        _, lb_param, ub_param = bci[1]      # return format: estimate, lower, upper
        return lb_param, ub_param
    end

end


function nswildbs_CI(idxs, Y, δ, L, mode,
                     tail=:both,
                     τ=1, num_boot=1000,
                     aggf=mean, rng=MersenneTwister(0))

    # fb = fourierseries(Float64, 8)
    fb = fourierseries(Float64, 0)
    nt = normalize_time(L, τ)
    ϕ(t) = fb(nt(t))

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B = get_coefs(Φ, ϕτ)

    if tail == :left
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, δ, num_boot, aggf, rng)
    elseif tail == :right
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, 1-δ, num_boot, aggf, rng)
    elseif tail == :both
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, [δ/2.0, 1-(δ/2.0)], num_boot, aggf, rng, mode)
    else
        println("ERROR tail: '$tail' not recognized. Returning NaN")
        return NaN
    end
end


#########################################
# Wild bootstrap-T stuff
# Algortithms 1,2, and 3 in the work by
#
# Y. Chandak, S. Jordan, G. Theocharous, M. White, and P. S. Thomas.
# Towards Safe Policy Improvement for Non-Stationary MDPs.
# In Advances in Neural Information Processing Systems, 2020.
########################################

function get_coefst(Φ, ϕτ)
    H = pinv(Φ' * Φ) * Φ'
    W = Φ * H
    ϕ = ϕτ * H
    C = I - W
    return W, ϕ, C
end

function get_preds_and_residual_t(Y, W, ϕ, C)
    Ŷ = W*Y
    y = ϕ * Ŷ
    x = C * Ŷ
    ξ = Y .- Ŷ
    Σ = ϕ * Diagonal(ξ.^2) * ϕ'
    v = mean(Σ)

    return y, v, x, ξ
end

function wildbst_eval(x, C, ξ, ϕ, σ)
    ξ̂ = ξ .* σ
    ξ̂2 = (x .+ C * ξ̂).^2
    Δŷ = mean(ϕ * ξ̂)
    Σ = ϕ * Diagonal(ξ̂2) * ϕ'
    v̂ = mean(Σ)
    return Δŷ / √v̂
end

function bst_CIs(p, v, bsamples, δ)
    return p - quantile(sort(bsamples), 1.0-δ, sorted=true) * √v
end

function bst_CIs(p, v, bsamples, δ::Array{T,1}) where {T}
    return p .- quantile(sort(bsamples), 1.0 .- δ, sorted=true) .* √v
end

function wildbst_CI(y, v, x, ξ, ϕ, C, δ, num_boot,rng)
    # bsamples = [wildbst_eval(x, C, ξ, ϕ, sign.(randn(rng, length(ξ)))) for i in 1:num_boot]
    bsamples = [wildbst_eval(x, C, ξ, ϕ, sign.(randn(length(ξ)))) for i in 1:num_boot] # Without RNG
    p = mean(y)
    # return p - quantile(sort(bsamples), 1.0-δ, sorted=true) * √v
    return bst_CIs(p, v, bsamples, δ)
end


function nswildbst_CI(idxs, Y, δ, L,
                        tail=:both,
                        τ=1, num_boot=1000,
                        aggf=mean, rng=MersenneTwister(0))

    fb = fourierseries(Float64, 4)
    nt = normalize_time(L, τ)
    ϕ(t) = fb(nt(t))

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B, C = get_coefst(Φ, ϕτ)

    if tail == :left
        return wildbst_CI(get_preds_and_residual_t(Y, A, B, C)..., B, C, δ, num_boot, rng)
    elseif tail == :right
        return wildbst_CI(get_preds_and_residual_t(Y, A, B, C)..., B, C, 1-δ, num_boot, rng)
    elseif tail == :both
        return wildbst_CI(get_preds_and_residual_t(Y, A, B, C)..., B, C, [δ/2.0, 1-(δ/2.0)], num_boot, rng)
    else
        println("ERROR tail: '$tail' not recognized. Returning NaN")
        return NaN
    end
    return
end



"""
create features for time series using basis function ϕ,
for observed time points x, and future time points tau
"""
function create_features(ϕ, x, τ)
    Φ = hcat(ϕ.(x)...)'
    ϕτ = hcat(ϕ.(τ)...)'
    return Φ, ϕτ
end


function fourierseries(::Type{T}, order::Int) where {T}
    C = collect(T, 0:order) .* π
    return x -> @. cos(C * x)
end

function normalize_time(L, τ)
    return x -> x / (L + τ)
end
