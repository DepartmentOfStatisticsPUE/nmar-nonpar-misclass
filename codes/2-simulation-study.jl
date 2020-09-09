

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
## coverage error ~ 80%
df.eta_cov = @. 1.5 * (df.x1 == 3) - 0.3 * (df.x2 == 0) 
df.pr_cov = @. exp(df.eta_cov) / (1 + exp(df.eta_cov))

## response error
df.eta_sel = @. 1.5 * (df.y == 2)
df.pr_sel = @. exp(df.eta_sel) / (1 + exp(df.eta_sel))
 
### observed data 
df.flag_cov = [rand(Bernoulli(i)) for i in df.pr_cov]
df.flag_sel = [rand(Bernoulli(i)) for i in df.pr_sel]

## save data 
savejdf("data/sim-data.jdf", df) ## ~18 mb

df.flag_obs = df.flag_sel .* df.flag_cov


## audit sample -- simple random sample (different size ? 1000 / 5000)
audit_sample = df[sample(findall(df.flag_obs), 2000),:]

sim_result = hcat(df.y |> freqtable |> prop,
                  df.y[df.flag_obs .==1] |> freqtable |> prop,
                  df.z2_y[df.flag_obs .==1] |> freqtable |> prop,
                  df.z3_y[df.flag_obs .==1] |> freqtable |> prop) 

setnames!(sim_result, ["true", "naive_y", "sim2_y", "sim3_y"], 2);
sim_result

## estimate probabilities
### estimate probs for x2 (binary)
model_z3_x2 = glm(@formula(z3_x2 ~ x2), audit_sample, Binomial(), LogitLink());
predict(model_z3_x2)

model_z3_x3 = fit(EconometricModel, @formula(z2_y ~ y), audit_sample);
model_z3_x3_pr = mapslices(softmax, fitted(model_z3_x3), dims =2)


## using all information (testing whether the method works)

res = nmar_npar(:flag_sel, :y, :y, [:x1, :x2], df)

hcat(
    freqtable(res.y, weights = res.m_hat .+ res.n) |> prop,
    freqtable(df.y) |> prop,
    freqtable(df.y[df.flag_obs .==1]) |> prop
)


## naive (without weighting)
## naive non-parametric 
## corrected non-parametric

l