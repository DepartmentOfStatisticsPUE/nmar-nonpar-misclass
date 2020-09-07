### simulation based on data from riddles et al.
## population data
Random.seed!(123);
strata_names = [11,12,21,22]
strata_sizes = 500 .* [50,100,150,200]
x = inverse_rle(strata_names, strata_sizes)
df = DataFrame(x = x)
df.x1 = ifelse.(SubString.(string.(df.x), 1, 1) .== "1", 1, 2)
df.x2 = ifelse.(SubString.(string.(df.x), 2, 2) .== "1", 1, 2)
df.y = vcat(
    wsample([1,2], [0.7,0.3], strata_sizes[1]),  
    wsample([1,2], [0.5,0.5], strata_sizes[2]), 
    wsample([1,2], [0.3,0.7], strata_sizes[3]),  
    wsample([1,2], [0.4,0.6], strata_sizes[4])
)
df.eta = -0.4 .* (df.x1 .== 2) .- 0.8 .* (df.y .== 2)
df.rho = 1 ./ (1 .+ exp.(df.eta))
