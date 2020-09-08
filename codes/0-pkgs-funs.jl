using Distributions
using Random
using DataFrames
using DataFramesMeta
using Plots
using Statistics
using StatsBase
using FreqTables
using HypothesisTests
using NamedArrays
using StatsModels
using GLM
using Econometrics
using StatsFuns
using JDF ## serialization

function vcramer(x)
    test_res = ChisqTest(x)
    statistic = sqrt(test_res.stat / (test_res.n*min(size(x,1)-1,size(x,2)-1)))
    return statistic
end



function nmar_npar(selection, target, calvars, totalvars, data::DataFrame)
    
    vars_all = unique(vcat([selection, calvars, totalvars, target]...))
    vars_cal_tot = unique(vcat([calvars, totalvars]...))
    vars_cal_tot = setdiff(vars_cal_tot, [target])

    df_sampl = by(data, vars_all, n = selection => length)
    df_sampl_obs = df_sampl[df_sampl[!, selection].== 1, :]
    df_sampl_obs = @transform(groupby(df_sampl_obs, vars_cal_tot), p_hat = :n/sum(:n))
    df_sampl_nonobs = by(df_sampl[df_sampl[!, selection] .== 0,:], vars_cal_tot, m = :n => sum)
    df_sampl_obs = leftjoin(df_sampl_obs, df_sampl_nonobs, on = vars_cal_tot)
    df_sampl_obs.O = 1
    O_start = df_sampl_obs.O

    for iter in 1:10000
        df_sampl_obs = @transform(groupby(df_sampl_obs, totalvars), m_hat = :m .* :p_hat .* :O / sum(:p_hat .* :O))
        df_sampl_obs = @transform(groupby(df_sampl_obs, calvars), O = sum(:m_hat) / sum(:n))
        dif = sum((O_start - df_sampl_obs.O).^2)   
        if (dif < sqrt(eps()))
            println("Converged on interation: ", iter, " with diff equal to ", dif)
            break
        end
        O_start = df_sampl_obs.O
    end

    return df_sampl_obs
end 
