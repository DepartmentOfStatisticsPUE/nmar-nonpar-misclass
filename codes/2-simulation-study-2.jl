

## population data
Random.seed!(123);
strata_names = [11,12,21,22,31,32]
strata_sizes = 2000 .* [50,100,150,200, 250, 300]
strata = inverse_rle(strata_names, strata_sizes)
df = DataFrame(strata=strata)
df.x1 = @. ifelse(SubString(string(df.strata), 1, 1) == "1", 1, 
            ifelse(SubString(string(df.strata), 1, 1) == "2", 2, 3))
df.x2 = @. ifelse(SubString(string(df.strata), 2, 2) == "1", 0, 1)
df.y = vcat(
    wsample([1,2,3], [0.7, 0.2, 0.1], strata_sizes[1]),  
    wsample([1,2,3], [0.4, 0.4, 0.2], strata_sizes[2]), 
    wsample([1,2,3], [0.2, 0.7, 0.1], strata_sizes[3]),  
    wsample([1,2,3], [0.4, 0.4, 0.2], strata_sizes[4]),
    wsample([1,2,3], [0.1, 0.6, 0.3], strata_sizes[5]),
    wsample([1,2,3], [0.2, 0.3, 0.5], strata_sizes[6])
)
categorical!(df, [:x1, :x2, :y])

## totals
x1_totals = freqtable(df.x1)
x2_totals = freqtable(df.x2)
x1_x2_totals = freqtable(df.x1, df.x2)

## create z variables
### sim1 
sim1_C_x1  = [0.9  0.05  0.05; 0.2  0.6  0.2; 0.2  0.1  0.7]
df.z1_x1 = 0
df.z1_x1[df.x1 .== 1] .= wsample([1, 2, 3], sim1_C_x1[1, :], sum(df.x1 .==1))
df.z1_x1[df.x1 .== 2] .= wsample([1, 2, 3], sim1_C_x1[2, :], sum(df.x1 .==2))
df.z1_x1[df.x1 .== 3] .= wsample([1, 2, 3], sim1_C_x1[3, :], sum(df.x1 .==3))

### sim2
sim2_C_y = [0.85  0.05  0.10; 0.10  0.75  0.15 ; 0.25  0.10  0.65]
df.z2_y = 0
df.z2_y[df.y .== 1] .= wsample([1, 2, 3], sim2_C_y[1, :], sum(df.y .==1))
df.z2_y[df.y .== 2] .= wsample([1, 2, 3], sim2_C_y[2, :], sum(df.y .==2))
df.z2_y[df.y .== 3] .= wsample([1, 2, 3], sim2_C_y[3, :], sum(df.y .==3))

### sim3
sim3_C_x1 = [0.85  0.05  0.10; 0.10  0.75  0.15; 0.25  0.10  0.65]
sim3_C_x2 = [0.9  0.1; 0.4  0.6]
sim3_C_y = [0.95  0.04  0.01; 0.05  0.85  0.10; 0.25  0.10  0.65]

df.z3_x1 = 0
df.z3_x2 = 0
df.z3_y = 0
df.z3_y[df.y .== 1] .= wsample([1, 2, 3], sim3_C_y[1, :], sum(df.y .==1))
df.z3_y[df.y .== 2] .= wsample([1, 2, 3], sim3_C_y[2, :], sum(df.y .==2))
df.z3_y[df.y .== 3] .= wsample([1, 2, 3], sim3_C_y[3, :], sum(df.y .==3))
df.z3_x1[df.x1 .== 1] .= wsample([1, 2, 3], sim3_C_x1[1, :], sum(df.x1 .==1))
df.z3_x1[df.x1 .== 2] .= wsample([1, 2, 3], sim3_C_x1[2, :], sum(df.x1 .==2))
df.z3_x1[df.x1 .== 3] .= wsample([1, 2, 3], sim3_C_x1[3, :], sum(df.x1 .==3))
df.z3_x2[df.x2 .== 0] .= wsample([0, 1], sim3_C_x2[1, :], sum(df.x2 .==0))
df.z3_x2[df.x2 .== 1] .= wsample([0, 1], sim3_C_x2[2, :], sum(df.x2 .==1))
df.z3_x2 = Array(df.z3_x2)
## change to categorical
categorical!(df, [:x1, :x2, :y, :z1_x1, :z2_y, :z3_x1, :z3_x2, :z3_y])


##  correlations 
vcramer(freqtable(df.y, df.x1))
vcramer(freqtable(df.y, df.x2))
vcramer(freqtable(df.x1, df.x2))

##### this is for simulation
## response error
df.eta_sel = @. 0.5 * (df.y == 3) - 1.5 * (df.x2 == 0) ## very bad classification

df.pr_sel = @. exp(df.eta_sel) / (1 + exp(df.eta_sel))
 
## dataframe to save results

df_results = DataFrame(
    iter=Int64[], cat = Int64[], 
    known=Float64[], 
    noerr=Float64[], 
    mis1a=Float64[], mis1b=Float64[], 
    mis2a=Float64[], mis2b=Float64[], 
    mis3a=Float64[], mis3b=Float64[], 
    naive=Float64[])

    
for b in 1:100
    Random.seed!(b);
    ### observed data 
    df.flag_sel = [rand(Bernoulli(i)) for i in df.pr_sel]
    ## audit sample from selected
    audit_sample = df[sample(findall(df.flag_sel), 2000),:]
    ### estimation of probabilities
    #model_z1_x1 = fit(LassoModel, @formula(x1  ~ z1_x1), audit_sample, ); ## multinomial
    #model_z2_y = fit(EconometricModel, @formula(y ~ z2_y ), audit_sample); ## multinomial
    #model_z3_x1 = fit(EconometricModel, @formula(x1 ~ z3_x1), audit_sample); ## multinomial
    #model_z3_x2 = fit(EconometricModel, @formula(x2 ~ z3_x2), audit_sample); ## binomial
    #model_z3_y = fit(EconometricModel, @formula(y ~ z3_y ), audit_sample); ## multinomial
    ### fitted probabilities
    #model_z1_x1_pr = DataFrame(mapslices(softmax, fitted(model_z1_x1), dims =2), ["z1_x1_1", "z1_x1_2", "z1_x1_3"]);
    #model_z2_y_pr = DataFrame(mapslices(softmax, fitted(model_z2_y), dims =2), ["z2_y_1", "z2_y_2", "z2_y_3"]);
    #model_z3_x1_pr = DataFrame(mapslices(softmax, fitted(model_z3_x1), dims =2), ["z3_x1_1", "z3_x1_2", "z3_x1_3"]);
    #model_z3_x2_pr = DataFrame(mapslices(softmax, fitted(model_z3_x2), dims =2), ["z3_x2_1", "z3_x2_2"]);
    #model_z3_y_pr = DataFrame(mapslices(softmax, fitted(model_z3_y), dims = 2), ["z3_y_1", "z3_y_2", "z3_y_3"]);    
    #audit_sampl_preds =  hcat(audit_sample, model_z1_x1_pr, model_z2_y_pr, model_z3_x1_pr, model_z3_x2_pr, model_z3_y_pr)
    ## aggregate data for models

    ## using all information (testing whether the method works)
    df_nocorr =  by(df, [:flag_sel, :x1, :x2, :y], n = :flag_sel => length)
    res_noerr = nmar_nonpar([:y, :x2], [:x1, :x2], [:flag_sel], [:y],  df_nocorr) 
    
    ## with errors in x1
    df_nocorr_1 =  by(df, [:flag_sel, :z1_x1, :x2, :y], n = :flag_sel => length)
    res_mis1a = nmar_nonpar([:y, :x2], [:z1_x1, :x2], [:flag_sel], [:y],  df_nocorr_1)
    
    ## correcting for errors in x1
    df_corr_1 = DataFrame(Array(prop(freqtable(audit_sample.z1_x1, audit_sample.x1), margins = 1)), ["1", "2", "3"])
    df_corr_1 = stack(df_corr_1, variable_name=:z1_x1, value_name=:prob)
    df_corr_1.x1 = repeat([1, 2, 3], 3)
    df_corr_1.z1_x1 = parse.(Int64, Array(df_corr_1.z1_x1))
    df_corrected = leftjoin(df_nocorr_1, df_corr_1, on = :z1_x1)
    df_corrected.n_prob = df_corrected.n .* df_corrected.prob
    df_corrected_model = by(df_corrected, [:flag_sel, :x1, :x2, :y], n = :n_prob => sum)
   
    res_mis1b = nmar_nonpar([:y, :x2], [:x1, :x2], [:flag_sel], [:y],  df_corrected_model)

    ## with errors in y
    df_nocorr_2 =  by(df, [:flag_sel, :x1, :x2, :z2_y], n = :flag_sel => length)
    res_mis2a = nmar_nonpar([:z2_y, :x2], [:x1, :x2], [:flag_sel], [:y],  df_nocorr_2) 

    df_corr_2 = DataFrame(Array(prop(freqtable(audit_sample.z2_y, audit_sample.y), margins = 1)), ["1", "2", "3"])
    df_corr_2 = stack(df_corr_2, variable_name=:z2_y, value_name=:prob)
    df_corr_2.y = repeat([1, 2, 3], 3)
    df_corr_2.z2_y = parse.(Int64, Array(df_corr_2.z2_y))
    df_corrected = leftjoin(df_nocorr_2, df_corr_2, on = :z2_y)
    df_corrected.n_prob = df_corrected.n .* df_corrected.prob
    df_corrected_model2 = by(df_corrected, [:flag_sel, :x1, :x2, :y], n = :n_prob => sum)
   
    res_mis2b = nmar_nonpar([:y, :x2], [:x1, :x2], [:flag_sel], [:y],  df_corrected_model2)

    ## with errors on all 
    df_nocorr_3 =  by(df, [:flag_sel, :z3_x1, :z3_x2, :z3_y], n = :flag_sel => length)
    res_mis3a = nmar_nonpar([:z3_y, :z3_x2], [:z3_x1, :z3_x2], [:flag_sel], [:y],  df_nocorr_3)

    df_corr_3_y = DataFrame(Array(prop(freqtable(audit_sample.z3_y, audit_sample.y), margins = 1)), ["1", "2", "3"])
    df_corr_3_y = stack(df_corr_3_y, variable_name=:z3_y, value_name=:prob_y)
    df_corr_3_y.y = repeat([1, 2, 3], 3)
    df_corr_3_y.z3_y = parse.(Int64, Array(df_corr_3_y.z3_y))
    df_corr_3_x1 = DataFrame(Array(prop(freqtable(audit_sample.z3_x1, audit_sample.x1), margins = 1)), ["1", "2", "3"])
    df_corr_3_x1 = stack(df_corr_3_x1, variable_name=:z3_x1, value_name=:prob_x1)
    df_corr_3_x1.x1 = repeat([1, 2, 3], 3)    
    df_corr_3_x1.z3_x1 = parse.(Int64, Array(df_corr_3_x1.z3_x1))
    df_corr_3_x2 = DataFrame(Array(prop(freqtable(audit_sample.z3_x2, audit_sample.x2), margins = 1)'), ["0", "1"])
    df_corr_3_x2 = stack(df_corr_3_x2, variable_name=:z3_x2, value_name=:prob_x2)
    df_corr_3_x2.x2 = repeat([0, 1], 2)   
    df_corr_3_x2.z3_x2 = parse.(Int64, Array(df_corr_3_x2.z3_x2))
    
    df_corrected = leftjoin(df_nocorr_3, df_corr_3_y, on = :z3_y)
    df_corrected = leftjoin(df_corrected, df_corr_3_x1, on = :z3_x1)
    df_corrected = leftjoin(df_corrected, df_corr_3_x2, on = :z3_x2)
    df_corrected.n_prob = df_corrected.n .* df_corrected.prob_y .* df_corrected.prob_x1 .* df_corrected.prob_x2
    df_corrected_model3 = by(df_corrected, [:flag_sel, :x1, :x2, :y], n = :n_prob => sum)
    
    res_mis3b = nmar_nonpar([:y, :x2], [:x1, :x2], [:flag_sel], [:y],  df_corrected_model3)

    sim_res = DataFrame(
        iter = repeat([b], length(unique(df.y))),
        cat = sort(unique(df.y)), 
        known = freqtable(df.y) |> prop,
        noerr = freqtable(res_noerr.y, weights =  res_noerr.m_hat .+ res_noerr.n ) |> prop,
        mis1a= freqtable(res_mis1a.y, weights = res_mis1a.m_hat .+ res_mis1a.n) |> prop,
        mis1b= freqtable(res_mis1b.y, weights = res_mis1b.m_hat .+ res_mis1b.n) |> prop,
        mis2a= freqtable(res_mis2a.z2_y, weights = res_mis2a.m_hat .+ res_mis2a.n) |> prop,
        mis2b= freqtable(res_mis2b.y, weights = res_mis2b.m_hat .+ res_mis2b.n) |> prop,
        mis3a = freqtable(res_mis3a.z3_y, weights = res_mis3a.m_hat .+ res_mis3a.n) |> prop,
        mis3b = freqtable(res_mis3b.y, weights = res_mis3b.m_hat .+ res_mis3b.n) |> prop,
        naive = freqtable(df.y[df.flag_sel .==1]) |> prop
        ) 
        append!(df_results, sim_res)
end

grr = groupby(df_results, :cat)
expected_values = combine(grr, valuecols(grr) .=> mean)
expected_bias = expected_values[:, 4:end] .- expected_values.known_mean
sum(abs.(Array(expected_bias)), dims = 1)

expected_variance = combine(grr, valuecols(grr) .=> var)

expected_mse = (Array(expected_bias.^2) .+ Array(expected_variance[:, 4:end])) .* 10000


## naive (without weighting)
## naive non-parametric 
## corrected non-parametric

## variables
df_plot = stack(df_results, [:known, :noerr, :mis1, :mis2, :mis3, :naive])
df_plot.variable = String.(df_plot.variable)
StatsPlots.boxplot(df_plot.variable, df_plot.value, group = df_plot.cat)

by(res_noerr, :y, x -> sum(x.n .* (1 .+ x.O))) 
