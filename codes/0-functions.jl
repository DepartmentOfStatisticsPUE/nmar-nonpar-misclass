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

function vcramer(x)
    test_res = ChisqTest(x)
    statistic = sqrt(test_res.stat / (test_res.n*min(size(x,1)-1,size(x,2)-1)))
    return statistic
end

