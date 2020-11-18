## using R packages
R"library(GJRM)"
R"library(ggplot2)"

## population data
Random.seed!(123);
strata_sizes = 40 .* [50, 100, 150, 200];
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

## parametric model for y
# Random.seed!(123);
# Σ = fill(0.5,  2, 2)
# Σ[diagind(Σ)] .= 1
# u_hat = rand(MvNormal([0, 0], Σ), sum(strata_sizes))
# df.y1 = @. ifelse(-1 + df.x1  + u_hat[1,:] > 0, 1, 0)
# df.y2 = @. ifelse(-1 + 2*df.x1  - 3*df.x2+ u_hat[2,:] > 0, 1, 0)
# @rput df
# R"gjrm_result_normal = gjrm(formula =  list(y1 ~  1 + x1, y ~ 1 + x1 + x2), 
# data = df, BivD = 'N', Model = 'BSS', margins = c('probit', 'probit'))"
# R"summary(gjrm_result_normal)"
# gjrm_res_norm = R"prev(gjrm_result_normal)"

## correlation
freqtable(df.x1, df.y) |> vcramer |> (x -> round(x, digits = 2))
freqtable(df.x2, df.y) |> vcramer |> (x -> round(x, digits = 2))
freqtable(df.x2, df.x1) |> vcramer |> (x -> round(x, digits = 2))




est_results = DataFrame(
        iter = Int64[],
        y = Int64[],
        y_true = Float64[],
        y_naive = Float64[],
        y_nonpar = Float64[],
        y_heck_n = Float64[],
        y_heck_gumb = Float64[],
        y_heck_clay = Float64[],
        y_heck_joe = Float64[],
        y_heck_gumb90= Float64[]
)

## gjrm package


for b in 1:200
    Random.seed!(b);
    println("======= ", b,  " =======")
    ## selection 
    df.η = @. -0.4 * (df.x1 == 1) - 0.8 * (df.y == 1);
    df.ρ = @. 1/(1 + exp(df.η));
    df.flag_sel = [rand(Bernoulli(i)) for i in df.ρ];
    df_nocorr =  by(df, [:flag_sel, :x1, :x2, :y], n = :flag_sel => length)
    res_mis1a = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_nocorr)

    @rput df
    R"gjrm_result_normal = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'N', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_gumb = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'G0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_clay = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'C0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_joe = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'J0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_gumb90 = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
       data = df, BivD = 'G90', Model = 'BSS', margins = c('logit', 'logit'))"
    
    gjrm_res_norm = R"prev(gjrm_result_normal)"
    gjrm_res_gumb = R"prev(gjrm_result_gumb)"
    gjrm_res_clay = R"prev(gjrm_result_clay)"
    gjrm_res_joe = R"prev(gjrm_result_joe)"
    gjrm_res_gumb90 = R"prev(gjrm_result_gumb90)"

    est_results_iter = DataFrame(
            iter = repeat([b], length(unique(df.y))),
            y = sort(unique(df.y)),
            y_true = freqtable(df.y) |> prop,
            y_naive = freqtable(df.y[df.flag_sel .== 1]) |> prop,
            y_nonpar = freqtable(res_mis1a.y, weights = res_mis1a.n .+ res_mis1a.m_hat) |> prop,
            y_heck_n = [1-gjrm_res_norm[1][2], gjrm_res_norm[1][2]],
            y_heck_gumb = [1-gjrm_res_gumb[1][2], gjrm_res_gumb[1][2]],
            y_heck_clay = [1-gjrm_res_clay[1][2], gjrm_res_clay[1][2]],
            y_heck_joe = [1-gjrm_res_joe[1][2], gjrm_res_joe[1][2]],
            y_heck_gumb90 = [1-gjrm_res_gumb90[1][2], gjrm_res_gumb90[1][2]]
    )
    
    append!(est_results,est_results_iter)
end



@pipe est_results |> 
    stack(_, Not([:iter, :y, :y_true])) |>
    @transform(_, bias = :y_true .- :value) |>
    groupby(_, [:y, :variable]) |>
    combine(_, :bias => mean => :bias, :value => Statistics.var => :var) |>
    @transform(_, mse = round.(:bias.^2 .+ :var, digits = 4),
                  bias = round.(:bias, digits = 4),
                  var = round.(:var, digits = 4)) |>
    @where(_, :y .== 1) |>
    select(_, Not(:y)) |>
    latexify(_, env=:table, latex=false)


## for plot
est_results_for_plot_sim_1 = @pipe est_results |> 
    stack(_, Not([:iter, :y, :y_true])) |>
    @transform(_, bias = :y_true .- :value) |>
    @where(_, :y .== 1) 

@rput est_results_for_plot_sim_1

R"wykres <- ggplot(est_results_for_plot_sim_1, aes(x=variable,y=bias)) + 
geom_boxplot() + labs(x='Estimator', y = 'Bias')+
geom_jitter(alpha = 0.1) + 
geom_hline(yintercept=0, color = 'red', linetype='dotted')"

R"ggsave(plot = wykres, file = 'sim-study-1.png', width=8, heigh = 6)"

## mar

est_results_mar = DataFrame(
        iter = Int64[],
        y = Int64[],
        y_true = Float64[],
        y_naive = Float64[],
        y_nonpar = Float64[],
        y_heck_n = Float64[],
        y_heck_gumb = Float64[],
        y_heck_clay = Float64[],
        y_heck_joe = Float64[],
        y_heck_gumb90= Float64[]
)

## gjrm package


for b in 1:200
    Random.seed!(b);
    println("======= ", b,  " =======")
    ## selection 
    df.η = @. -0.4 * (df.x1 == 1);
    df.ρ = @. 1/(1 + exp(df.η));
    df.flag_sel = [rand(Bernoulli(i)) for i in df.ρ];
    df_nocorr =  by(df, [:flag_sel, :x1, :x2, :y], n = :flag_sel => length)
    res_mis1a = nmar_nonpar([:x1], [:x1, :x2], [:flag_sel], [:y],  df_nocorr)

    @rput df
    R"gjrm_result_normal = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'N', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_gumb = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'G0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_clay = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'C0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_joe = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'J0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_gumb90 = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
       data = df, BivD = 'G90', Model = 'BSS', margins = c('logit', 'logit'))"
    
    gjrm_res_norm = R"prev(gjrm_result_normal)"
    gjrm_res_gumb = R"prev(gjrm_result_gumb)"
    gjrm_res_clay = R"prev(gjrm_result_clay)"
    gjrm_res_joe = R"prev(gjrm_result_joe)"
    gjrm_res_gumb90 = R"prev(gjrm_result_gumb90)"

    est_results_iter_mar = DataFrame(
            iter = repeat([b], length(unique(df.y))),
            y = sort(unique(df.y)),
            y_true = freqtable(df.y) |> prop,
            y_naive = freqtable(df.y[df.flag_sel .== 1]) |> prop,
            y_nonpar = freqtable(res_mis1a.y, weights = res_mis1a.n .+ res_mis1a.m_hat) |> prop,
            y_heck_n = [1-gjrm_res_norm[1][2], gjrm_res_norm[1][2]],
            y_heck_gumb = [1-gjrm_res_gumb[1][2], gjrm_res_gumb[1][2]],
            y_heck_clay = [1-gjrm_res_clay[1][2], gjrm_res_clay[1][2]],
            y_heck_joe = [1-gjrm_res_joe[1][2], gjrm_res_joe[1][2]],
            y_heck_gumb90 = [1-gjrm_res_gumb90[1][2], gjrm_res_gumb90[1][2]]
    )
    
    append!(est_results_mar,est_results_iter_mar)
end


@pipe est_results_mar |> 
    stack(_, Not([:iter, :y, :y_true])) |>
    @transform(_, bias = :y_true .- :value) |>
    groupby(_, [:y, :variable]) |>
    combine(_, :bias => mean => :bias, :value => Statistics.var => :var) |>
    @transform(_, mse = round.(:bias.^2 .+ :var, digits = 4),
                  bias = round.(:bias, digits = 4),
                  var = round.(:var, digits = 4)) |>
    @where(_, :y .== 1) |>
    select(_, Not(:y)) |>
    latexify(_, env=:table, latex=false)

est_results_for_plot_sim_2 = @pipe est_results_mar |> 
    stack(_, Not([:iter, :y, :y_true])) |>
    @transform(_, bias = :y_true .- :value) |>
    @where(_, :y .== 1) 

@rput est_results_for_plot_sim_2

R"wykres <- ggplot(est_results_for_plot_sim_2, aes(x=variable,y=bias)) + 
geom_boxplot() + labs(x='Estimator', y = 'Bias')+
geom_jitter(alpha = 0.1) + 
geom_hline(yintercept=0, color = 'red', linetype='dotted')"

R"ggsave(plot = wykres, file = 'sim-study-2.png', width=8, heigh = 6)"

## nmar 2 

est_results_nmar2 = DataFrame(
        iter = Int64[],
        y = Int64[],
        y_true = Float64[],
        y_naive = Float64[],
        y_nonpar = Float64[],
        y_heck_n = Float64[],
        y_heck_gumb = Float64[],
        y_heck_clay = Float64[],
        y_heck_joe = Float64[],
        y_heck_gumb90= Float64[]
)

## gjrm package


for b in 1:200
    Random.seed!(b);
    println("======= ", b,  " =======")
    ## selection 
    df.η = @. - 0.8 * (df.y == 1);
    df.ρ = @. 1/(1 + exp(df.η));
    df.flag_sel = [rand(Bernoulli(i)) for i in df.ρ];
    df_nocorr =  by(df, [:flag_sel, :x1, :x2, :y], n = :flag_sel => length)
    res_mis1a = nmar_nonpar([:y, :x1], [:x1, :x2], [:flag_sel], [:y],  df_nocorr)

    @rput df
    R"gjrm_result_normal = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'N', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_gumb = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'G0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_clay = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'C0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_joe = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
    data = df, BivD = 'J0', Model = 'BSS', margins = c('logit', 'logit'))"
    R"gjrm_result_gumb90 = gjrm(formula =  list(flag_sel ~ -1 + x1, y ~ -1 + x1 + x2), 
       data = df, BivD = 'G90', Model = 'BSS', margins = c('logit', 'logit'))"
    
    gjrm_res_norm = R"prev(gjrm_result_normal)"
    gjrm_res_gumb = R"prev(gjrm_result_gumb)"
    gjrm_res_clay = R"prev(gjrm_result_clay)"
    gjrm_res_joe = R"prev(gjrm_result_joe)"
    gjrm_res_gumb90 = R"prev(gjrm_result_gumb90)"

    est_results_iter_nmar2 = DataFrame(
            iter = repeat([b], length(unique(df.y))),
            y = sort(unique(df.y)),
            y_true = freqtable(df.y) |> prop,
            y_naive = freqtable(df.y[df.flag_sel .== 1]) |> prop,
            y_nonpar = freqtable(res_mis1a.y, weights = res_mis1a.n .+ res_mis1a.m_hat) |> prop,
            y_heck_n = [1-gjrm_res_norm[1][2], gjrm_res_norm[1][2]],
            y_heck_gumb = [1-gjrm_res_gumb[1][2], gjrm_res_gumb[1][2]],
            y_heck_clay = [1-gjrm_res_clay[1][2], gjrm_res_clay[1][2]],
            y_heck_joe = [1-gjrm_res_joe[1][2], gjrm_res_joe[1][2]],
            y_heck_gumb90 = [1-gjrm_res_gumb90[1][2], gjrm_res_gumb90[1][2]]
    )
    
    append!(est_results_nmar2,est_results_iter_nmar2)
end

@pipe est_results_nmar2 |> 
    stack(_, Not([:iter, :y, :y_true])) |>
    @transform(_, bias = :y_true .- :value) |>
    groupby(_, [:y, :variable]) |>
    combine(_, :bias => mean => :bias, :value => Statistics.var => :var) |>
    @transform(_, mse = round.(:bias.^2 .+ :var, digits = 4),
                  bias = round.(:bias, digits = 4),
                  var = round.(:var, digits = 4)) |>
    @where(_, :y .== 1) |>
    select(_, Not(:y)) |>
    latexify(_, env=:table, latex=false)

est_results_for_plot_sim_3 = @pipe est_results_nmar2 |> 
    stack(_, Not([:iter, :y, :y_true])) |>
    @transform(_, bias = :y_true .- :value) |>
    @where(_, :y .== 1) 

@rput est_results_for_plot_sim_3

R"wykres <- ggplot(est_results_for_plot_sim_3, aes(x=variable,y=bias)) + 
geom_boxplot() + labs(x='Estimator', y = 'Bias')+
geom_jitter(alpha = 0.1) + 
geom_hline(yintercept=0, color = 'red', linetype='dotted')"

R"ggsave(plot = wykres, file = 'sim-study-3.png', width=8, heigh = 6)"


