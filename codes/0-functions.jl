using Distributions
using Random
using DataFramesMeta
using Plots
using Statistics
using StatsBase
using FreqTables


function vcramer(x)
    test_res = ChisqTest(x)
    statistic = sqrt(test_res.stat / (test_res.n*min(size(x,1)-1,size(x,2)-1)))
    return statistic
end