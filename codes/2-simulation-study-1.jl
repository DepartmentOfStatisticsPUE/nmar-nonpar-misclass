## population data
### simulation study based on Riddles et al.
Random.seed!(123);
strata_sizes = 5000 .* [50, 100, 150, 200];
strata_names = [11,12,21,22];
strata = inverse_rle(strata_names, strata_sizes);
df = DataFrame(strata=strata);
df.x1 = @. ifelse(SubString(string(df.strata), 1, 1) == "1", 0, 1);
df.x2 = @. ifelse(SubString(string(df.strata), 2, 2) == "1", 0, 1);
df.y = vcat(
    wsample([0,1], [0.7, 0.3], strata_sizes[1]),  
    wsample([0,1], [0.5, 0.5], strata_sizes[2]), 
    wsample([0,1], [0.3, 0.7], strata_sizes[3]),  
    wsample([0,1], [0.4, 0.6], strata_sizes[4]),
);

## correlations
freqtable(df.x1, df.y) |> vcramer
freqtable(df.x2, df.y) |> vcramer


### misclassification matrices for x1, x2 and y

df.x1_star  = 0
df.x1_star[df.x1 .== 0] .= rand(Bernoulli(0.25), sum(df.x1 .== 0)) 
df.x1_star[df.x1 .== 1] .= rand(Bernoulli(0.95), sum(df.x1 .== 1)) 
prop(freqtable(df.x1, df.x1_star), margins = 1)

prop(freqtable(df.x1, df.x1_star), margins = 1) |> (x -> round.(x, digits=2)) |> lap

df.x2_star = 0
df.x2_star[df.x2 .== 0] .= rand(Bernoulli(0.15), sum(df.x2 .== 0)) 
df.x2_star[df.x2 .== 1] .= rand(Bernoulli(0.80), sum(df.x2 .== 1)) 
prop(freqtable(df.x2, df.x2_star), margins = 1)

prop(freqtable(df.x2, df.x2_star), margins = 1) |> (x -> round.(x, digits=2)) |> lap


df.y_star = 0
df.y_star[df.y .== 0] .= rand(Bernoulli(0.15), sum(df.y .== 0)) 
df.y_star[df.y .== 1] .= rand(Bernoulli(0.90), sum(df.y .== 1)) 
prop(freqtable(df.y, df.y_star), margins = 1)
prop(freqtable(df.y, df.y_star), margins = 1) |> (x -> round.(x, digits=2)) |> lap

df.x1_x2 = string.(df.x1) .* string.(df.x2)
df.x1_x2_star = string.(df.x1_star) .* string.(df.x2_star)
df.y_x1_x2 = df.x1_x2 .* string.(df.y)
df.y_x1_x2_star = df.x1_x2_star .* string.(df.y_star)

### save est_results

est_results = DataFrame(
        iter = Int64[],
        y = Int64[],
        y_true = Float64[],
        y_naive = Float64[],
        y_corr  = Float64[],
        y_corr_cal  = Float64[],
        y_err_x1 = Float64[],
        y_err_x1_cal = Float64[],
        #y_err_x1_cor = Float64[],
        y_err_x2 = Float64[],
        y_err_x2_cal = Float64[],
        #y_err_x2_cor = Float64[],
        y_err_x1_x2 = Float64[],
        y_err_x1_x2_cal = Float64[],
        #y_err_x1_x2_cor = Float64[],
        y_err_y = Float64[],
        y_err_y_cal = Float64[],
        #y_err_y_cor = Float64[],
        y_err_y_x1_x2 = Float64[],
        y_err_y_x1_x2_cal = Float64[]
        #y_err_y_x1_x2_cor = Float64[],
        #y_err_y_x1_x2_cor2 = Float64[]
    )

for b in 1:500

    Random.seed!(b);
    ### selection
    df.η = @. -0.4 * (df.x1 == 1) - 0.8 * (df.y == 1);
    #df.η = @. -0.4 * (df.x1 == 1);
    df.ρ = @. 1/(1 + exp(df.η));
    df.flag_sel = [rand(Bernoulli(i)) for i in df.ρ];
    totals_x1x2 = sum(Array(df[:,[:x1, :x2]]), dims = 1)
    d = ones(sum(df.flag_sel))
    y  = df.y[df.flag_sel .== 1]
    y_star = df.y_star[df.flag_sel .== 1] 
    ## no errors
    df_nocorr =  by(df, [:flag_sel, :x1, :x2, :y], n = :flag_sel => length)
    res_mis1a = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_nocorr)
    
    ## linear calib
    res_mis1a_cal = lin_calib(Array(df[df.flag_sel .== 1, [:x1, :x2]]),d, totals_x1x2)   

    ## with error in x1 - 
    df_err_x1 =  by(df, [:flag_sel, :x1_star, :x2, :y], n = :flag_sel => length)
    res_err_x1 = nmar_nonpar([:y, :x1_star], [:x1_star, :x2], [:flag_sel], [:y],  df_err_x1)
    res_err_x1_cal = lin_calib(Array(df[df.flag_sel .== 1, [:x1_star, :x2]]),d, totals_x1x2) 
    ## with error in x2 - 
    df_err_x2 =  by(df, [:flag_sel, :x1, :x2_star, :y], n = :flag_sel => length)
    res_err_x2 = nmar_nonpar([:y, :x1], [:x1, :x2_star], [:flag_sel], [:y],  df_err_x2)
    res_err_x2_cal = lin_calib(Array(df[df.flag_sel .== 1, [:x1, :x2_star]]),d, totals_x1x2) 

    ## with error in x1 and x2
    df_err_x1_x2 =  by(df, [:flag_sel, :x1_star, :x2_star, :y], n = :flag_sel => length)
    res_err_x1_x2 = nmar_nonpar([:y, :x1_star], [:x1_star, :x2_star], [:flag_sel], [:y],  df_err_x1_x2)
    res_err_x1_x2_cal = lin_calib(Array(df[df.flag_sel .== 1, [:x1_star, :x2_star]]),d, totals_x1x2) 

    ## with error in y, not in x1, x2
    df_err_y =  by(df, [:flag_sel, :x1, :x2, :y_star], n = :flag_sel => length)
    res_err_y = nmar_nonpar([:y_star, :x1], [:x1, :x2], [:flag_sel], [:y_star],  df_err_y)

    ## with errors in y, x1, x2
    df_err_y_x1_x2 =  by(df, [:flag_sel, :x1_star, :x2_star, :y_star], n = :flag_sel => length)
    res_err_y_x1_x2 = nmar_nonpar([:y_star, :x1_star], [:x1_star, :x2_star], [:flag_sel], [:y_star],  df_err_y_x1_x2)
    
    est_results_iter = DataFrame(
        iter = repeat([b], length(unique(df.y))),
        y = sort(unique(df.y)),
        y_true = freqtable(df.y) |> prop,
        y_naive = freqtable(df.y[df.flag_sel .== 1]) |> prop,
        y_corr  = freqtable(res_mis1a.y, weights = res_mis1a.n .+ res_mis1a.m_hat) |> prop,
        y_corr_cal = [ sum(res_mis1a_cal .* (y .== 0)) / sum(res_mis1a_cal); sum(res_mis1a_cal .* (y .== 1)) / sum(res_mis1a_cal)],

        ## errors in x1, x2
        y_err_x1 = freqtable(res_err_x1.y, weights = res_err_x1.n .+ res_err_x1.m_hat) |> prop,
        y_err_x1_cal = [ sum(res_err_x1_cal .* (y .== 0)) / sum(res_err_x1_cal); sum(res_err_x1_cal .* (y .== 1)) / sum(res_err_x1_cal)],
        #y_err_x1_cor = freqtable(res_err_x1_cor.y, weights = res_err_x1_cor.n .+ res_err_x1_cor.m_hat) |> prop,
        y_err_x2 = freqtable(res_err_x2.y, weights = res_err_x2.n .+ res_err_x2.m_hat) |> prop,
        y_err_x2_cal = [ sum(res_err_x2_cal .* (y .== 0)) / sum(res_err_x2_cal); sum(res_err_x2_cal .* (y .== 1)) / sum(res_err_x2_cal)],
        #y_err_x2_cor = freqtable(res_err_x2_cor.y, weights = res_err_x2_cor.n .+ res_err_x2_cor.m_hat) |> prop,
        y_err_x1_x2 = freqtable(res_err_x1_x2.y, weights = res_err_x1_x2.n .+ res_err_x1_x2.m_hat) |> prop,
        y_err_x1_x2_cal = [ sum(res_err_x1_x2_cal .* (y .== 0)) / sum(res_err_x1_x2_cal); sum(res_err_x1_x2_cal .* (y .== 1)) / sum(res_err_x1_x2_cal)],
        #y_err_x1_x2_cor = freqtable(res_err_x1_x2_cor.y, weights = res_err_x1_x2_cor.n .+ res_err_x1_x2_cor.m_hat) |> prop,
        ### errors in y
        y_err_y = freqtable(res_err_y.y_star, weights = res_err_y.n .+ res_err_y.m_hat) |> prop,
        y_err_y_cal = [ sum(res_mis1a_cal .* (y_star .== 0)) / sum(res_mis1a_cal); sum(res_mis1a_cal .* (y_star .== 1)) / sum(res_mis1a_cal)],
        #y_err_y_cor = freqtable(res_err_y_cor.y, weights = res_err_y_cor.n .+ res_err_y_cor.m_hat) |> prop,
        ### errors in all variables
        y_err_y_x1_x2 = freqtable(res_err_y_x1_x2.y_star, weights = res_err_y_x1_x2.n .+ res_err_y_x1_x2.m_hat) |> prop,
        y_err_y_x1_x2_cal = [ sum(res_err_x1_x2_cal .* (y_star .== 0)) / sum(res_err_x1_x2_cal); sum(res_err_x1_x2_cal .* (y_star .== 1)) / sum(res_err_x1_x2_cal)]
        #y_err_y_x1_x2_cor = freqtable(res_err_y_x1_x2_cor.y, weights = res_err_y_x1_x2_cor.n .+ res_err_y_x1_x2_cor.m_hat) |> prop,
        #y_err_y_x1_x2_cor2 = freqtable(res_err_y_x1_x2_cor2.y, weights = res_err_y_x1_x2_cor2.n .+ res_err_y_x1_x2_cor2.m_hat) |> prop
    )
    append!(est_results,est_results_iter)
end

est_gr = groupby(est_results, :y)
expected_values = combine(est_gr, valuecols(est_gr) .=> mean)
expected_bias = expected_values[:, 4:end] .- expected_values.y_true_mean  
expected_variance = combine(est_gr, valuecols(est_gr) .=> var)


summary_results = DataFrame(estim = names(expected_bias), 
         bias = Array(expected_bias[2,:]), 
         var = Array(expected_variance[2,4:end]),
         mse = Array(expected_bias[2,:]).^2 .+  Array(expected_variance[2,4:end]))

summary_results.mse = summary_results.mse .* 100