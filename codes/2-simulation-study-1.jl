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
        y_err_x1 = Float64[],
        y_err_x1_cor = Float64[],
        y_err_x2 = Float64[],
        y_err_x2_cor = Float64[],
        y_err_x1_x2 = Float64[],
        y_err_x1_x2_cor = Float64[],
        y_err_y = Float64[],
        y_err_y_cor = Float64[],
        y_err_y_x1_x2 = Float64[],
        y_err_y_x1_x2_cor = Float64[],
        y_err_y_x1_x2_cor2 = Float64[]
    )

for b in 1:500

    Random.seed!(b);
    ### selection
    df.η = @. -0.4 * (df.x1 == 1) - 0.8 * (df.y == 1);
    df.ρ = @. 1/(1 + exp(df.η));
    df.flag_sel = [rand(Bernoulli(i)) for i in df.ρ];

    ## audit sample -- maybe based on size / stratified?
    audit_sample = df[sample(findall(df.flag_sel), 5000),:]
    model_x1 = glm(@formula(x1 ~ x1_star + x2_star + y), audit_sample, Bernoulli())
    model_x2 = glm(@formula(x2 ~ x2_star + x1_star + y), audit_sample, Bernoulli())
    model_y = glm(@formula(y ~ y_star + x1 + x2), audit_sample, Bernoulli())
    model_y_x1_x2 = glm(@formula(y ~ y_star + x1_star + x2_star), audit_sample, Bernoulli())

    ## no errors
    df_nocorr =  by(df, [:flag_sel, :x1, :x2, :y], n = :flag_sel => length)
    res_mis1a = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_nocorr)

    ## with error in x1 - 
    df_err_x1 =  by(df, [:flag_sel, :x1_star, :x2, :y], n = :flag_sel => length)
    res_err_x1 = nmar_nonpar([:y, :x1_star], [:x1_star, :x2], [:flag_sel], [:y],  df_err_x1)
    
    ## correct errors in x1 based on audit sample
    df_err_x1.x1_1 = predict(model_x1, df_err_x1)
    df_err_x1.x1_0 = 1 .- df_err_x1.x1_1
    df_err_x1_cor = stack(df_err_x1, [:x1_1, :x1_0], value_name = :p, variable_name = :x1)
    df_err_x1_cor.x1 = parse.(Int, replace.(df_err_x1_cor.x1, "x1_" => ""))
    df_err_x1_cor.n = df_err_x1_cor.n .* df_err_x1_cor.p
    df_err_x1_cor = by(df_err_x1_cor, [:flag_sel, :x1, :x2, :y], n = :n => sum)
    res_err_x1_cor = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_err_x1_cor)

    ## with error in x2 - 
    df_err_x2 =  by(df, [:flag_sel, :x1, :x2_star, :y], n = :flag_sel => length)
    res_err_x2 = nmar_nonpar([:y, :x1], [:x1, :x2_star], [:flag_sel], [:y],  df_err_x2)

    ## correct errirs in x2 based on audit sample
    df_err_x2.x2_1 = predict(model_x2, df_err_x2)
    df_err_x2.x2_0 = 1 .- df_err_x2.x2_1
    df_err_x2_cor = stack(df_err_x2, [:x2_1, :x2_0], value_name = :p, variable_name = :x2)
    df_err_x2_cor.x2 = parse.(Int, replace.(df_err_x2_cor.x2, "x2_" => ""))
    df_err_x2_cor.n = df_err_x2_cor.n .* df_err_x2_cor.p
    df_err_x2_cor = by(df_err_x2_cor, [:flag_sel, :x1, :x2, :y], n = :n => sum)
    res_err_x2_cor = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_err_x2_cor)

    ## with error in x1 and x2
    df_err_x1_x2 =  by(df, [:flag_sel, :x1_star, :x2_star, :y], n = :flag_sel => length)
    res_err_x1_x2 = nmar_nonpar([:y, :x1_star], [:x1_star, :x2_star], [:flag_sel], [:y],  df_err_x1_x2)

    ## correct errors errors in x1, x2
    df_err_x1_x2_cor = leftjoin(df_err_x1_x2, pp_x1_x2, on = [:x1_star, :x2_star])
    df_err_x1_x2_cor.n = df_err_x1_x2_cor.n .* df_err_x1_x2_cor.p_mis
    df_err_x1_x2_cor = by(df_err_x1_x2_cor, [:flag_sel, :x1, :x2, :y], n = :n => sum)
    res_err_x1_x2_cor = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_err_x1_x2_cor)

    ## with error in y, not in x1, x2
    df_err_y =  by(df, [:flag_sel, :x1, :x2, :y_star], n = :flag_sel => length)
    res_err_y = nmar_nonpar([:y_star, :x1], [:x1, :x2], [:flag_sel], [:y_star],  df_err_y)

    ## correct error in y, not in x1, x2
    df_err_y.y_1 = predict(model_y, df_err_y)
    df_err_y.y_0 = 1 .- df_err_y.y_1
    df_err_y_cor = stack(df_err_y, [:y_1, :y_0], value_name = :p, variable_name = :y)
    df_err_y_cor.y = parse.(Int, replace.(df_err_y_cor.y, "y_" => ""))
    df_err_y_cor.n = df_err_y_cor.n .* df_err_y_cor.p
    df_err_y_cor = by(df_err_y_cor, [:flag_sel, :x1, :x2, :y], n = :n => sum)
    res_err_y_cor = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_err_y_cor)

    ## with errors in y, x1, x2
    df_err_y_x1_x2 =  by(df, [:flag_sel, :x1_star, :x2_star, :y_star], n = :flag_sel => length)
    res_err_y_x1_x2 = nmar_nonpar([:y_star, :x1_star], [:x1_star, :x2_star], [:flag_sel], [:y_star],  df_err_y_x1_x2)
    
    ### correct errors in y, x1, x2
    df_err_y_x1_x2_cor = leftjoin(df_err_y_x1_x2, pp_y_x1_x2, on = [:y_star, :x1_star, :x2_star])
    df_err_y_x1_x2_cor.n = df_err_y_x1_x2_cor.n .* df_err_y_x1_x2_cor.p_mis
    df_err_y_x1_x2_cor = by(df_err_y_x1_x2_cor, [:flag_sel, :x1, :x2, :y], n = :n => sum)
    res_err_y_x1_x2_cor = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_err_y_x1_x2_cor)

    ## correct error only in y, even if x1, x2 are measured with errors
    df_err_y_x1_x2_cor2 = leftjoin(df_err_y_x1_x2, pp_y, on = :y_star)
    df_err_y_x1_x2_cor2.n = df_err_y_x1_x2_cor2.n .* df_err_y_x1_x2_cor2.p_mis
    df_err_y_x1_x2_cor2 = by(df_err_y_x1_x2_cor2, [:flag_sel, :x1_star, :x2_star, :y], n = :n => sum)
    res_err_y_x1_x2_cor2 = nmar_nonpar([:y, :x1_star], [:x1_star, :x2_star], [:flag_sel], [:y],  df_err_y_x1_x2_cor2)
    
    est_results_iter = DataFrame(
        iter = repeat([b], length(unique(df.y))),
        y = sort(unique(df.y)),
        y_true = freqtable(df.y) |> prop,
        y_naive = freqtable(df.y[df.flag_sel .== 1]) |> prop,
        y_corr  = freqtable(res_mis1a.y, weights = res_mis1a.n .+ res_mis1a.m_hat) |> prop,
        ## errors in x1, x2
        y_err_x1 = freqtable(res_err_x1.y, weights = res_err_x1.n .+ res_err_x1.m_hat) |> prop,
        y_err_x1_cor = freqtable(res_err_x1_cor.y, weights = res_err_x1_cor.n .+ res_err_x1_cor.m_hat) |> prop,
        y_err_x2 = freqtable(res_err_x2.y, weights = res_err_x2.n .+ res_err_x2.m_hat) |> prop,
        y_err_x2_cor = freqtable(res_err_x2_cor.y, weights = res_err_x2_cor.n .+ res_err_x2_cor.m_hat) |> prop,
        y_err_x1_x2 = freqtable(res_err_x1_x2.y, weights = res_err_x1_x2.n .+ res_err_x1_x2.m_hat) |> prop,
        y_err_x1_x2_cor = freqtable(res_err_x1_x2_cor.y, weights = res_err_x1_x2_cor.n .+ res_err_x1_x2_cor.m_hat) |> prop,
        ### errors in y
        y_err_y = freqtable(res_err_y.y_star, weights = res_err_y.n .+ res_err_y.m_hat) |> prop,
        y_err_y_cor = freqtable(res_err_y_cor.y, weights = res_err_y_cor.n .+ res_err_y_cor.m_hat) |> prop,
        ### errors in all variables
        y_err_y_x1_x2 = freqtable(res_err_y_x1_x2.y_star, weights = res_err_y_x1_x2.n .+ res_err_y_x1_x2.m_hat) |> prop,
        y_err_y_x1_x2_cor = freqtable(res_err_y_x1_x2_cor.y, weights = res_err_y_x1_x2_cor.n .+ res_err_y_x1_x2_cor.m_hat) |> prop,
        y_err_y_x1_x2_cor2 = freqtable(res_err_y_x1_x2_cor2.y, weights = res_err_y_x1_x2_cor2.n .+ res_err_y_x1_x2_cor2.m_hat) |> prop
    )

    append!(est_results,est_results_iter)

end

est_results

est_results_2000 = est_results
est_results_10000 = est_results

est_gr = groupby(est_results, :y)
expected_values = combine(est_gr, valuecols(est_gr) .=> mean)
expected_bias = expected_values[:, 4:end] .- expected_values.y_true_mean  

stack(expected_values, Not(:y))

sum(abs.(Array(expected_bias)), dims = 1)[:]

expected_variance = combine(est_gr, valuecols(est_gr) .=> var)


DataFrame(estim = names(expected_bias), 
         bias = Array(expected_bias[2,:]), 
         var = Array(expected_variance[2,4:end]),
         mse = Array(expected_bias[2,:]).^2 .+  Array(expected_variance[2,4:end]))

