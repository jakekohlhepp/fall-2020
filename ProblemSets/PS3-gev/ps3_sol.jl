using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using CategoricalArrays
using Tables
using StatsModels
cd("C:/Users/jakek/Documents/econ_phd/structural_modeling_course/forked_class/ProblemSets/PS3-gev/")

## 1.
df = CSV.read("nlsw88w.csv")
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation

# make cat 8 the base category.
function mlogit(alpha, X, d,Z)
	Zbar = Z .- Z[:,8]
	u = reduce(hcat,[X*alpha[(3*i-2):(3*i)] .+ Zbar[:,i] .*alpha[22] for i in 1:7])
	help = reduce(hcat,[d.==i for i in 1:7])
	loglike = sum(help .*u) - sum(log.(1 .+sum(exp.(u), dims=2)))
	return -loglike
end

# starting values - set intercepts to fraction present in data.
starting = Vector(prop(freqtable(df, :occupation)))
starting = repeat(starting[Not(8)], inner = 3)
starting[Not((1:7) .*3 .-2)] .= 0
# one extra row for the wage coef
starting = append!(starting, 0)

beta_hat_mlogit = optimize(b -> mlogit(b, X, y,Z),starting, BFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
res = reduce(hcat,[vec(beta_hat_mlogit.minimizer)[(3*i-2):(3*i)] for i in 1:7])
res = [res [vec(beta_hat_mlogit.minimizer)[22] 0 0]']
CSV.write("q1.csv", convert(DataFrame,res))
# gamma hat represents how much people value wages relative to non-wage amenities. it also represents how sensitive job choice is to relative wage

## 3.
# construct nests.
df.nest = (3 .* (df.occupation .>7) .+  Int.(df.occupation .<=3) .+ 2 .* (3 .< df.occupation .<=7) )
 
# make other (3) the base.
# denote the lambdas as the last coef in the set.

function nlogit(alpha, X, d,Z)
	Zbar = Z .- Z[:,8]
	u = reduce(hcat,[exp.((X*alpha[1:3].*(i<=3)+ X*alpha[5:7].*(3<i<=7).+ Zbar[:,i] .*alpha[9]) ./ (alpha[4]*(i<=3)+alpha[8]*(3<i<=7))) for i in 1:8])
	u[:, 8] .= 1
	wc_denom = sum(u[:,1:3], dims=2) .^alpha[4]
	bc_denom = sum(u[:,4:7], dims=2) .^alpha[8]
	u[:, 1:3] = (u[:, 1:3] .* wc_denom .^(-1)) ./ (1 .+ bc_denom .+wc_denom)
	u[:, 4:7] = u[:, 4:7] .* bc_denom .^(-1) ./ (1 .+ bc_denom .+wc_denom)
	u[:, 8] = u[:, 8] ./ (1 .+ bc_denom .+wc_denom)
	help = reduce(hcat,[d.==i for i in 1:8])
	loglike = sum(log.( help .*u ))
	return -loglike
end

# starting values - set intercepts to fraction present in data.
starting = Vector(prop(freqtable(df, :nest)))
starting = repeat(starting[Not(3)], inner = 4)
starting[Not((1:2) .*4 .-3)] .= 0
starting .= 0
starting[[4 8]] .= 1
# one extra row for the wage coef
starting = append!(starting, 0)
beta_hat_mlogit = optimize(b -> nlogit(b, X, y,Z),starting, BFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
