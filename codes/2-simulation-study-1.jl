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

df.z1 = 0
df.z1[df.x1 .== 0] .= rand(Bernoulli(0.25), sum(df.x1 .== 0)) 
df.z1[df.x1 .== 1] .= rand(Bernoulli(0.95), sum(df.x1 .== 1)) 
prop(freqtable(df.x1, df.z1), margins = 1)

df.z2 = 0
df.z2[df.x2 .== 0] .= rand(Bernoulli(0.15), sum(df.x2 .== 0)) 
df.z2[df.x2 .== 1] .= rand(Bernoulli(0.80), sum(df.x2 .== 1)) 
prop(freqtable(df.x2, df.z2), margins = 1)

df.zy = 0
df.zy[df.y .== 0] .= rand(Bernoulli(0.15), sum(df.y .== 0)) 
df.zy[df.y .== 1] .= rand(Bernoulli(0.90), sum(df.y .== 1)) 
prop(freqtable(df.y, df.zy), margins = 1)


### selection
df.η = @. -0.4 * (df.x1 == 1) - 0.8 * (df.y == 1);
df.ρ = @. 1/(1 + exp(df.η));
df.flag_sel = [rand(Bernoulli(i)) for i in df.ρ];
[mean(df.flag_sel) sum(df.flag_sel)]

## audit sample
audit_sample = df[sample(findall(df.flag_sel), 2000),:]

### modelling with GLM
model_x1 = glm(@formula(x1 ~ z1), audit_sample, Bernoulli())
model_x2 = glm(@formula(x2 ~ z2), audit_sample, Bernoulli())
model_y = glm(@formula(y ~ zy), audit_sample, Bernoulli())

## instead of stratified sampling we will just use ρ to mimic big data
