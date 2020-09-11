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
using GLM
using StatsFuns
using JDF ## serialization
using RData
using RegressionTables

function vcramer(x)
    test_res = ChisqTest(x)
    statistic = sqrt(test_res.stat / (test_res.n*min(size(x,1)-1,size(x,2)-1)))
    return statistic
end



function nmar_npar(X, Z, data::DataFrame, maxiter=10000, tol = sqrt(eps()))
    
    data_temp = data
    O_start = data_temp.O

    for iter in 1:maxiter
        data_temp = @transform(groupby(data_temp, Z), m_hat = :m .* :p_hat .* :O / sum(:p_hat .* :O))
        data_temp = @transform(groupby(data_temp, X), O = sum(:m_hat) / sum(:n))
        dif = sum((O_start - data_temp.O).^2)   
        if (dif < tol)
            println("Converged on interation: ", iter, " with diff equal to ", dif)
            break
        end
        O_start = data_temp.O
    end

    return data_temp
end 


