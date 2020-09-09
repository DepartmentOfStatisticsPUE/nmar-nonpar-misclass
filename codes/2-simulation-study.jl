

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

## change to categorical
categorical!(df, [:x1, :x2, :y, :z1_x1, :z2_y, :z3_x1, :z3_x2, :z3_y])


##  correlations 
vcramer(freqtable(df.y, df.x1))
vcramer(freqtable(df.y, df.x2))
vcramer(freqtable(df.x1, df.x2))

##### this is for simulation
## response error
df.eta_sel = @. 1.5 * (df.y == 3) - 0.5 * (df.x2 == 1)
df.pr_sel = @. exp(df.eta_sel) / (1 + exp(df.eta_sel))
 
## dataframe to save results

df_results = DataFrame(
    iter=Int64[], cat = Int64[], 
    known=Float64[], noerr=Float64[], 
    mis1=Float64[], mis2=Float64[], mis3=Float64[], 
    naive=Float64[])

for b in 1:200
    Random.seed!(b)
    ### observed data 
    df.flag_sel = [rand(Bernoulli(i)) for i in df.pr_sel]
    ## audit sample from selected
    audit_sample = df[sample(findall(df.flag_sel), 2000),:]
    
    ### estimation of probabilities
    model_z3_x1 = glm(@formula(z3_x1 ~ x1), audit_sample, Binomial(), LogitLink());
    model_z3_x2 = glm(@formula(z3_x2 ~ x2), audit_sample, Binomial(), LogitLink());
    model_z3_x2 = glm(@formula(z3_x2 ~ x2), audit_sample, Binomial(), LogitLink());
    

    ## using all information (testing whether the method works)
    res_noerr = nmar_npar(:flag_sel, :y, [:y, :x2], [:x1, :x2], df)
    res_mis1 = nmar_npar(:flag_sel, :y, [:y, :x2], [:z1_x1, :x2], df)
    res_mis2 = nmar_npar(:flag_sel, :z2_y, [:z2_y, :x2], [:x1, :x2], df)
    res_mis3 = nmar_npar(:flag_sel, :z3_y, [:z3_y, :z3_x2], [:z3_x1, :z3_x2], df)

    sim_res = DataFrame(
        iter = [b,b,b],
        cat = sort(unique(df.y)), 
        known = freqtable(df.y) |> prop,
        #noerr = freqtable(res_noerr.y, weights = res_noerr.m_hat .+ res_noerr.n) |> prop,
        noerr = freqtable(res_noerr.y, weights =  res_noerr.m_hat .+ res_noerr.n ) |> prop,
        mis1= freqtable(res_mis1.y, weights = res_mis1.m_hat .+ res_mis1.n) |> prop,
        mis2= freqtable(res_mis2.z2_y, weights = res_mis2.m_hat .+ res_mis2.n) |> prop,
        mis3 = freqtable(res_mis3.z3_y, weights = res_mis3.m_hat .+ res_mis3.n) |> prop,
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

