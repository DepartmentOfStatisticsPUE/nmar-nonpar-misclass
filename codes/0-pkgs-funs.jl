using Distributions
using Random
using DataFrames
using DataFramesMeta
using Plots
using Statistics
using StatsBase
using StatsPlots
using FreqTables
using HypothesisTests
using NamedArrays
using StatsModels
using Econometrics
using MLJ
using MLJLinearModels
using StatsFuns
using JDF ## serialization
using RData
using RegressionTables
using SplitApplyCombine
using LatexPrint

function vcramer(x)
    test_res = ChisqTest(x)
    statistic = sqrt(test_res.stat / (test_res.n*min(size(x,1)-1,size(x,2)-1)))
    return statistic
end


function nmar_nonpar(X::Array{Symbol,1}, Z::Array{Symbol,1}, sel::Array{Symbol,1}, target::Array{Symbol,1}, 
                     data::DataFrame; maxiter=20000, tol = sqrt(eps()))::DataFrame
    
    #vars_all = unique(vcat([X, Z, sel]...))
    vars_XZ_only = setdiff(unique(vcat([X, Z]...)), target)

    #df_sampl = by(data, vars_all, n = sel[1] => length)
    df_sampl = data
    df_sampl_obs = df_sampl[ df_sampl[:, sel[1]] .== 1, :]
    df_sampl_obs = @transform(groupby(df_sampl_obs, vars_XZ_only), p_hat = :n/sum(:n))
    df_sampl_nonobs = by(df_sampl[ df_sampl[:, sel[1]] .== 0,:], vars_XZ_only, m = :n => sum)
    df_sampl_obs = leftjoin(df_sampl_obs, df_sampl_nonobs, on = vars_XZ_only)
    df_sampl_obs.O = 1
    O_start = df_sampl_obs.O
    iter_m = 0
    for iter in 1:maxiter
        df_sampl_obs = @transform(groupby(df_sampl_obs, Z), m_hat = :m .* :p_hat .* :O / sum(:p_hat .* :O))
        df_sampl_obs = @transform(groupby(df_sampl_obs, X), O = sum(:m_hat) / sum(:n))
        dif = sum((O_start - df_sampl_obs.O).^2)   
        if (dif < tol)
            println("Converged on interation: ", iter, " with diff equal to ", dif)
            break
        end
        O_start = df_sampl_obs.O
        iter_m = iter
    end
    return df_sampl_obs
end 


## linear calibration

function lin_calib(X, d, T̂ₓ)
    @assert size(X)[1] == size(d)[1] "X and d have different number of rows"
    @assert size(X)[2] == size(T̂ₓ)[2] "X and T̂ₓ have different number of columns"
    @assert sum(X) > 0 "All elements of X are zero"
    @assert all(sum(X, dims = 1) .> 0) "One of X column has only zero elements"
    @assert all(T̂ₓ .> 0) "Some total is equal to zero"
    D = Diagonal(d);
    w = d + D*X*inv(X'D*X)*(T̂ₓ - d'X)';
    return(w)
end
